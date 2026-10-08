"""Audio analysis: Discogs-EffNet embedding plus classification heads.

One pass over the audio yields everything Next Track needs about a track:

- a 2560-dim embedding (mean + std of the 1280-dim EffNet patch activations)
  used for "sounds like" similarity,
- per-track probabilities from the Essentia heads (party, aggressive, relaxed,
  electronic, voice, happy, sad) that define the suggestion directions,
- the top Discogs styles from the 400-class genre head, for display,
- optionally BPM and key, for audio that rekordbox hasn't analysed.

Essentia and TensorFlow can segfault on malformed files, so the models live in
a ``spawn`` subprocess (same pattern as ``scripts/ingest_music.py``). A crash
only kills the child; :class:`Analyser` restarts it and carries on.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Optional

from . import paths

log = logging.getLogger(__name__)

# feature name -> (head stem, positive class label)
HEADS = {
    "party":      ("mood_party", "party"),
    "aggressive": ("mood_aggressive", "aggressive"),
    "relaxed":    ("mood_relaxed", "relaxed"),
    "electronic": ("mood_electronic", "electronic"),
    "voice":      ("voice_instrumental", "voice"),
    "happy":      ("mood_happy", "happy"),
    "sad":        ("mood_sad", "sad"),
}
GENRE_STEM = "genre_discogs400"

# Analyse at most this much audio, taken from the middle of the file. Enough to
# characterise a track, and it caps the cost of hour-long mixes.
MAX_SECONDS = 240
EMB_SR = 16000
RHYTHM_SR = 44100


def _head_path(stem: str, ext: str) -> str:
    return str(paths.models_dir() / f"{stem}-discogs-effnet-1.{ext}")


def _middle(audio, sr: int, seconds: int):
    n = int(seconds * sr)
    if len(audio) <= n:
        return audio
    start = (len(audio) - n) // 2
    return audio[start:start + n]


def _worker_loop(effnet_path: str, task_q, result_q):
    """Child process: load every model once, then analyse files on request."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    try:
        import numpy as np
        import essentia
        essentia.log.warningActive = False
        essentia.log.infoActive = False
        import essentia.standard as es

        backbone = es.TensorflowPredictEffnetDiscogs(
            graphFilename=effnet_path, output="PartitionedCall:1")
        heads, pos = {}, {}
        for name, (stem, label) in HEADS.items():
            heads[name] = es.TensorflowPredict2D(
                graphFilename=_head_path(stem, "pb"), output="model/Softmax")
            classes = json.load(open(_head_path(stem, "json")))["classes"]
            pos[name] = classes.index(label)
        genre_meta = json.load(open(_head_path(GENRE_STEM, "json")))
        genre_classes = genre_meta["classes"]
        genre_head = es.TensorflowPredict2D(
            graphFilename=_head_path(GENRE_STEM, "pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0")
        backbone(np.zeros(5 * EMB_SR, dtype=np.float32))  # warm-up
        result_q.put(("ready", None))
    except Exception as exc:  # pragma: no cover - startup failure path
        result_q.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return

    import queue as _queue
    parent = os.getppid()
    while True:
        try:
            task = task_q.get(timeout=2)
        except _queue.Empty:
            # If the backend was killed (app quit, reload) we'd be orphaned
            # holding the models in memory. Leave with it.
            if os.getppid() != parent:
                break
            continue
        if task is None:
            break
        path, want_rhythm = task
        try:
            if want_rhythm:
                full = es.MonoLoader(filename=path, sampleRate=RHYTHM_SR,
                                     resampleQuality=4)()
                full = _middle(full, RHYTHM_SR, MAX_SECONDS)
                audio = es.Resample(inputSampleRate=RHYTHM_SR,
                                    outputSampleRate=EMB_SR, quality=4)(full)
            else:
                full = None
                audio = es.MonoLoader(filename=path, sampleRate=EMB_SR,
                                      resampleQuality=4)()
                duration = len(audio) / EMB_SR
                audio = _middle(audio, EMB_SR, MAX_SECONDS)
            if len(audio) < 3 * EMB_SR:
                result_q.put(("error", "audio shorter than 3 seconds"))
                continue

            patches = np.array(backbone(audio))
            if patches.ndim != 2 or not np.isfinite(patches).all():
                result_q.put(("error", f"bad embedding {patches.shape}"))
                continue
            emb = np.concatenate([patches.mean(axis=0), patches.std(axis=0)])

            head_out = {}
            for name, model in heads.items():
                pred = np.array(model(patches))
                head_out[name] = round(float(pred[:, pos[name]].mean()), 4)

            g = np.array(genre_head(patches)).mean(axis=0)
            top = np.argsort(g)[::-1][:5]
            genres = [[genre_classes[i], round(float(g[i]), 4)] for i in top]

            out = {
                "emb": emb.astype(np.float32).tolist(),
                "heads": head_out,
                "genres": genres,
            }
            if want_rhythm:
                out["duration"] = None
                window = _middle(full, RHYTHM_SR, 90)
                bpm, *_ = es.RhythmExtractor2013(method="multifeature")(window)
                bpm = float(bpm)
                # Fold into the dance-music range: 63 BPM is really 126.
                while bpm and bpm < 85:
                    bpm *= 2
                while bpm > 190:
                    bpm /= 2
                key, scale, strength = es.KeyExtractor(profileType="edma")(window)
                out["bpm"] = round(bpm, 1)
                out["key"] = f"{key}{'m' if scale == 'minor' else ''}"
                out["key_strength"] = round(float(strength), 3)
            else:
                out["duration"] = round(duration, 1)
            result_q.put(("ok", out))
        except Exception as exc:
            result_q.put(("error", f"{type(exc).__name__}: {exc}"))


class AnalyserUnavailable(RuntimeError):
    pass


class Analyser:
    """Thread-safe front end to the analysis subprocess. Starts lazily."""

    def __init__(self):
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._proc = None
        self._task_q = None
        self._result_q = None

    def _start(self):
        effnet = paths.effnet_model()
        if effnet is None:
            raise AnalyserUnavailable(
                "Discogs-EffNet model not found. Set ESSENTIA_MODEL_PATH.")
        self._task_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop, args=(str(effnet), self._task_q, self._result_q),
            daemon=True)
        self._proc.start()
        try:
            status, info = self._result_q.get(timeout=120)
        except Exception:
            status, info = "fatal", "timed out loading models"
        if status != "ready":
            self._kill()
            raise AnalyserUnavailable(f"Analyser failed to start: {info}")
        log.info("Next Track analyser ready (EffNet %s)", effnet)

    def _kill(self):
        try:
            if self._proc is not None:
                self._proc.terminate()
                self._proc.join(timeout=5)
        except Exception:
            pass
        self._proc = None

    def analyse(self, path: str | Path, want_rhythm: bool = False,
                timeout: int = 180) -> Optional[dict]:
        """Analyse one file. Returns the feature dict, or None if it failed."""
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._start()
            self._task_q.put((str(path), want_rhythm))
            try:
                status, result = self._result_q.get(timeout=timeout)
            except Exception:
                log.warning("Analysis timed out for %s; restarting worker", path)
                self._kill()
                return None
            if status == "ok":
                return result
            log.warning("Analysis failed for %s: %s", path, result)
            if not self._proc.is_alive():
                self._kill()
            return None

    def shutdown(self):
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                self._task_q.put(None)
                self._proc.join(timeout=10)
            self._kill()
