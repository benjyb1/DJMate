"""Audio analysis: Discogs-EffNet embedding plus classification heads.

One pass over the audio yields everything Next Track needs about a track:

- a 2560-dim embedding (mean + std of the 1280-dim EffNet patch activations)
  used for "sounds like" similarity,
- per-track probabilities from the Essentia heads (party, aggressive, relaxed,
  electronic, voice, happy, sad) that define the suggestion directions,
- the top Discogs styles from the 400-class genre head, for display,
- mean activations of the first and last minute (intro/outro), for judging
  how one track's ending sits against another's start,
- optionally BPM and key, for audio that rekordbox hasn't analysed.

Only the parts that are analysed get decoded: ffmpeg seeks to the middle four
minutes, the first minute and the last minute. Modes:

- "full": everything (library builds),
- "quick": the middle two minutes plus the outro (a track on air right now,
  a dropped file or a mic clip, where speed matters),
- "edges": intro and outro only (tracks analysed before those existed),
- "moods": the mood/theme labels only (likewise).

Essentia and TensorFlow can segfault on malformed files, so the models live in
a ``spawn`` subprocess (same pattern as ``scripts/ingest_music.py``). A crash
only kills the child; :class:`Analyser` restarts it and carries on.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import subprocess
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
# Optional multi-label mood/theme head (56 labels incl. dark, deep, energetic,
# groovy, heavy, space). Used when its files are present in models/.
MOOD_STEM = "mtg_jamendo_moodtheme"

# Analyse at most this much audio, taken from the middle of the file. Enough to
# characterise a track, and it caps the cost of hour-long mixes.
MAX_SECONDS = 240
EMB_SR = 16000
RHYTHM_SR = 44100


def _head_path(stem: str, ext: str) -> str:
    return str(paths.models_dir() / f"{stem}-discogs-effnet-1.{ext}")


EDGE_SECONDS = 60
QUICK_SECONDS = 120
FFMPEG = shutil.which("ffmpeg") or next(
    (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)), None)
FFPROBE = shutil.which("ffprobe") or next(
    (p for p in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe") if os.path.exists(p)), None)


class _Loader:
    """Reads just the parts of a file the analysis needs.

    With ffmpeg, each segment is a seek plus a short decode, so a seven-minute
    track costs a few seconds of audio rather than the whole file. Without
    ffmpeg it falls back to decoding the whole file once with Essentia and
    slicing.
    """

    def __init__(self, path: str, duration: Optional[float], es):
        self.path, self.es = path, es
        self._full: dict[int, object] = {}
        self.channels = None
        probed = self._probe()
        self.duration = duration or probed

    def _probe(self) -> Optional[float]:
        if FFPROBE:
            try:
                out = subprocess.run(
                    [FFPROBE, "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "stream=channels:format=duration",
                     "-of", "default=nw=1:nk=1", self.path],
                    capture_output=True, text=True, timeout=30)
                vals = [v for v in out.stdout.split() if v and v != "N/A"]
                self.channels = int(vals[0])
                return float(vals[1])
            except Exception:
                pass
        a = self._decode_all(EMB_SR)
        return len(a) / EMB_SR

    def _decode_all(self, sr: int):
        if sr not in self._full:
            self._full[sr] = self.es.MonoLoader(filename=self.path, sampleRate=sr,
                                                resampleQuality=4)()
        return self._full[sr]

    def segment(self, start: float, seconds: float, sr: int):
        import numpy as np
        if FFMPEG and self.channels:
            # Downmix as Essentia's MonoLoader does, a plain average of the
            # channels. ffmpeg's own "-ac 1" is 3 dB louder, and EffNet is
            # level-sensitive, so the fingerprints would drift.
            n = self.channels
            mix = ([] if n == 1 else
                   ["-af", "pan=mono|c0=" + "+".join(f"{1 / n:.6f}*c{i}" for i in range(n))])
            try:
                proc = subprocess.run(
                    [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{start:.3f}",
                     "-t", f"{seconds:.3f}", "-i", self.path, "-vn", *mix, "-ac", "1",
                     "-ar", str(sr), "-f", "f32le", "-"],
                    capture_output=True, timeout=120)
                if proc.returncode == 0 and proc.stdout:
                    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
            except Exception:
                pass
        a = self._decode_all(sr)
        i = int(start * sr)
        return a[i:i + int(seconds * sr)]


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
        mood_head, mood_classes = None, []
        if os.path.exists(_head_path(MOOD_STEM, "pb")) and os.path.exists(_head_path(MOOD_STEM, "json")):
            meta = json.load(open(_head_path(MOOD_STEM, "json")))
            mood_classes = meta["classes"]
            schema = meta.get("schema", {})
            out_name = next((o["name"] for o in schema.get("outputs", [])
                             if o.get("output_purpose") == "predictions"), "model/Sigmoid")
            in_name = (schema.get("inputs") or [{}])[0].get("name", "model/Placeholder")
            mood_head = es.TensorflowPredict2D(graphFilename=_head_path(MOOD_STEM, "pb"),
                                               input=in_name, output=out_name)
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
        path, want_rhythm, mode, duration = task
        try:
            loader = _Loader(path, duration, es)
            duration = loader.duration
            if not duration or duration < 3:
                result_q.put(("error", "audio shorter than 3 seconds"))
                continue

            def embed(start, seconds):
                a = loader.segment(start, seconds, EMB_SR)
                if len(a) < 3 * EMB_SR:
                    return None
                p = np.array(backbone(a))
                return p if p.ndim == 2 and np.isfinite(p).all() else None

            def moods_of(patches):
                if mood_head is None:
                    return None
                m = np.array(mood_head(patches)).mean(axis=0)
                return {c: round(float(v), 4) for c, v in zip(mood_classes, m)}

            if mode == "moods":
                # Only the mood/theme labels (tracks analysed before the
                # mood model was added).
                patches = embed(max(0.0, (duration - MAX_SECONDS) / 2), MAX_SECONDS)
                if patches is None:
                    result_q.put(("error", "couldn't embed the audio"))
                else:
                    result_q.put(("ok", {"moods": moods_of(patches)}))
                continue

            # Intro and outro: how the track starts and ends, for judging how
            # this track's outro sits against another's intro. "quick" (a track
            # on air right now) skips the intro: only its outro matters for
            # what comes next, and every second counts.
            intro = None if mode == "quick" else embed(0, EDGE_SECONDS)
            outro = embed(max(0.0, duration - EDGE_SECONDS), EDGE_SECONDS)
            edges = {
                "intro": intro.mean(axis=0).astype(np.float32).tolist() if intro is not None else None,
                "outro": outro.mean(axis=0).astype(np.float32).tolist() if outro is not None else None,
                "duration": round(duration, 1),
            }
            if mode == "edges":
                result_q.put(("ok", edges))
                continue

            window = QUICK_SECONDS if mode == "quick" else MAX_SECONDS
            mid_start = max(0.0, (duration - window) / 2)
            patches = embed(mid_start, window)
            if patches is None:
                result_q.put(("error", "couldn't embed the audio"))
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
                "moods": moods_of(patches),
                **edges,
            }
            if want_rhythm:
                window = loader.segment(max(0.0, (duration - 90) / 2), 90, RHYTHM_SR)
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
                timeout: int = 180, mode: str = "full",
                duration: Optional[float] = None) -> Optional[dict]:
        """Analyse one file. Returns the feature dict, or None if it failed.

        mode="edges" only computes the intro/outro embeddings (cheap, for
        tracks analysed before those existed)."""
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._start()
            self._task_q.put((str(path), want_rhythm, mode, duration))
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
