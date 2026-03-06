"""
ingest_music.py — Continuous music library ingestion pipeline
──────────────────────────────────────────────────────────────
Drop in your entire mixing folder. This script will:

  1. Recursively scan for new audio files (skips anything already in the DB)
  2. Read title/artist from file tags (mutagen)
  3. Extract BPM (librosa primary, Essentia fallback) + key (Essentia)
  4. Compute Discogs-EffNet embeddings (mean+std pooling)
  5. Upload to Supabase (tracks + track_labels tables)
  6. Upload audio file to Supabase Storage → set audio_url on the track
  7. Compute MFCC fingerprint from local file → upsert to track_features
  8. Run auto_tagger.py on the newly added tracks
  9. Re-run 3d_coordinator.py to update UMAP 3-D positions for the frontend

Usage:
  python scripts/ingest_music.py /path/to/mixing/folder
  python scripts/ingest_music.py /path/to/mixing/folder --dry-run
  python scripts/ingest_music.py /path/to/mixing/folder --skip-auto-tag
  python scripts/ingest_music.py /path/to/mixing/folder --skip-3d
  python scripts/ingest_music.py /path/to/mixing/folder --skip-audio-upload
  python scripts/ingest_music.py /path/to/mixing/folder --skip-fingerprints
  python scripts/ingest_music.py /path/to/mixing/folder --force-reembed

Model path resolution (in order):
  1. ESSENTIA_MODEL_PATH env var
  2. <project-root>/models/discogs-effnet-bs64.pb
  3. Sibling worktree (dev fallback)
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# Suppress TensorFlow / MLIR C++ noise before Essentia loads the TF runtime
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # ERROR only
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestor")

# ── Constants ─────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
COVER_BUCKET     = "album-covers"
AUDIO_BUCKET     = "audio-files"

AUDIO_MIME_MAP = {
    ".mp3":  "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a":  "audio/mp4",
    ".mp4":  "audio/mp4",
    ".ogg":  "audio/ogg",
    ".wav":  "audio/wav",
    ".aiff": "audio/aiff",
    ".aif":  "audio/aiff",
}

# MFCC fingerprint constants (must match compute_audio_fingerprints.py + LiveMode.jsx)
FP_SR       = 22050
FP_N_FFT    = 2048
FP_N_MELS   = 40
FP_N_MFCC   = 13
FP_FMIN     = 20.0
FP_FMAX     = 8000.0
FP_OFFSET   = 15.0
FP_DURATION = 90.0

# Project root = parent of this script's directory (scripts/../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locations to search for the Essentia EffNet model
_MODEL_CANDIDATES = [
    Path(os.getenv("ESSENTIA_MODEL_PATH", "")),   # env var (may be empty)
    PROJECT_ROOT / "models" / "discogs-effnet-bs64.pb",
    # dev fallback — worktree copy
    PROJECT_ROOT / ".claude" / "worktrees" / "lucid-stonebraker" /
    "Scripts" / "Backend" / "OlderFiles" / "Models" / "discogs-effnet-bs64.pb",
]


# ── Model helpers ─────────────────────────────────────────────────────────────

def _find_model() -> Optional[Path]:
    for candidate in _MODEL_CANDIDATES:
        if candidate and candidate.is_file():
            return candidate
    return None


def _embedding_worker_loop(model_path: str, task_q, result_q):
    """Persistent subprocess: loads the model once, then processes filepaths.

    Runs in an isolated ``spawn`` subprocess so SIGBUS / segfaults from
    MonoLoader or the TensorFlow graph only kill this child — the ingestor
    automatically restarts the worker and continues with the next track.
    """
    # TF noise suppression must also be set in the child (spawn = fresh env)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "ERROR"

    try:
        import numpy as np
        from essentia.standard import TensorflowPredictEffnetDiscogs, MonoLoader

        model = TensorflowPredictEffnetDiscogs(
            graphFilename=model_path,
            output="PartitionedCall:1",
        )
        # Warmup — fail fast before touching any real files
        silence    = np.zeros(5 * 16000, dtype=np.float32)
        warmup_out = np.array(model(silence))
        if warmup_out.ndim != 2 or warmup_out.shape[1] == 0:
            result_q.put(("fatal", f"Warmup output shape {warmup_out.shape}"))
            return
        result_q.put(("ready", tuple(warmup_out.shape)))

    except Exception as exc:
        result_q.put(("fatal", str(exc)))
        return

    # ── Main task loop ────────────────────────────────────────────────────────
    while True:
        filepath_str = task_q.get()
        if filepath_str is None:          # shutdown sentinel
            break
        try:
            audio = MonoLoader(
                filename=filepath_str,
                sampleRate=16000,
                resampleQuality=4,
            )()
            if len(audio) < 16000:
                result_q.put(("skip", "audio too short"))
                continue
            raw  = np.array(model(audio))
            if raw.ndim != 2:
                result_q.put(("error", f"unexpected embedding shape {raw.shape}"))
                continue
            mean = np.mean(raw, axis=0)
            std  = np.std(raw,  axis=0)
            emb  = np.concatenate([mean, std]).flatten().astype(np.float32)
            if np.isnan(emb).any() or np.isinf(emb).any() or np.all(emb == 0):
                result_q.put(("error", "nan/inf/zeros in embedding"))
                continue
            result_q.put(("ok", emb.tolist()))
        except Exception as exc:
            result_q.put(("error", str(exc)))


class EmbeddingWorker:
    """Manages a persistent embedding subprocess that survives per-file crashes.

    MonoLoader (Essentia/FFmpeg) can segfault on malformed audio files.
    Keeping the model in a subprocess means only the child dies; the ingestor
    restarts the worker and continues with the next track.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._ctx       = multiprocessing.get_context("spawn")
        self._proc      = None
        self._task_q    = None
        self._result_q  = None
        self._emb_dim   = None
        self._start()

    def _start(self):
        self._task_q   = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_embedding_worker_loop,
            args=(self.model_path, self._task_q, self._result_q),
            daemon=True,
        )
        self._proc.start()
        # Wait up to 90 s for model load + warmup
        try:
            status, info = self._result_q.get(timeout=90)
        except Exception:
            status, info = "fatal", "timeout waiting for worker startup"
        if status != "ready":
            self._proc.terminate()
            self._proc.join()
            raise RuntimeError(f"Embedding worker failed to start: {info}")
        self._emb_dim = info[1] * 2       # mean + std doubles the dim
        log.info(f"✓ Embedding worker ready — patch shape {info}, "
                 f"embedding dim = {self._emb_dim}")

    def _restart(self):
        log.warning("  Embedding worker crashed — restarting …")
        try:
            self._proc.terminate()
            self._proc.join(timeout=5)
        except Exception:
            pass
        self._start()

    def compute(self, filepath: Path, timeout: int = 180) -> Optional[list]:
        """Compute the embedding for one file. Returns a list or None on error."""
        if not self._proc.is_alive():
            self._restart()
        self._task_q.put(str(filepath))
        try:
            status, result = self._result_q.get(timeout=timeout)
        except Exception:
            log.warning("  Embedding timed out — restarting worker")
            self._restart()
            return None
        if status == "ok":
            return result
        if status == "skip":
            log.warning(f"  Embedding skipped: {result}")
        else:
            log.error(f"  Embedding error: {result}")
        if not self._proc.is_alive():
            self._restart()
        return None

    def shutdown(self):
        try:
            if self._proc and self._proc.is_alive():
                self._task_q.put(None)   # sentinel
                self._proc.join(timeout=10)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join()
        except Exception:
            pass


# ── Audio discovery ───────────────────────────────────────────────────────────

def find_audio_files(folder: Path) -> list[Path]:
    found = []
    for root, _, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                found.append(Path(root) / name)
    found.sort()
    return found


# ── Tag reading ───────────────────────────────────────────────────────────────

_ARTIST_TITLE_RE = re.compile(r"^(.+?)\s+-\s+(.+)$")
_UNKNOWN_ARTIST_VALUES = {
    "unknown", "unknown artist", "various", "various artists", "n/a", "none",
}


def read_tags(filepath: Path) -> tuple[str, str]:
    """Return (title, artist) from audio file tags. Falls back to filename.

    Uses mutagen.File(easy=True) — the universal reader that works across
    MP3, FLAC, M4A, WAV, AIFF, OGG etc. without format-specific branching.

    Also detects the 'Artist - Title' pattern when artist is unknown and
    splits it out so new tracks are named correctly from the start.
    """
    title, artist = None, None
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(str(filepath), easy=True)
        if audio is not None and audio.tags is not None:
            t = audio.tags.get("title")
            a = audio.tags.get("artist")
            if t:
                title  = str(t[0]) if isinstance(t, list) else str(t)
            if a:
                artist = str(a[0]) if isinstance(a, list) else str(a)
    except Exception as exc:
        log.debug(f"Tag read failed for {filepath.name}: {exc}")

    title  = title  or filepath.stem
    artist = artist or "Unknown"

    # If artist is unknown/blank, try to split "Artist - Title" out of either
    # the tag title or the filename stem (whichever matches first).
    # e.g. tag='Z@P-Saved' won't match (no spaces), but filename 'Z@P - Saved'
    # will, so we fall back to the stem.
    if artist.strip().lower() in _UNKNOWN_ARTIST_VALUES:
        m = _ARTIST_TITLE_RE.match(title) or _ARTIST_TITLE_RE.match(filepath.stem)
        if m:
            artist = m.group(1).strip()
            title  = m.group(2).strip()
            log.debug(f"  Parsed 'Artist - Title' → artist={artist!r} title={title!r}")

    return title, artist


# ── BPM + key extraction ──────────────────────────────────────────────────────

def extract_bpm_key(filepath: Path) -> tuple[Optional[float], Optional[str]]:
    bpm, key_str = None, None

    # BPM: librosa primary
    try:
        import librosa
        y, sr = librosa.load(str(filepath), duration=60, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        if 40 <= tempo <= 250:
            bpm = round(tempo, 2)
            log.debug(f"  BPM (librosa): {bpm}")
    except Exception as exc:
        log.debug(f"  librosa BPM failed: {exc}")

    # BPM: Essentia fallback
    if bpm is None:
        try:
            import essentia.standard as es
            audio = es.MonoLoader(filename=str(filepath), sampleRate=44100)()
            extractor = es.RhythmExtractor2013(method="multifeature")
            essentia_bpm, *_ = extractor(audio)
            essentia_bpm = float(essentia_bpm)
            if 40 <= essentia_bpm <= 250:
                bpm = round(essentia_bpm, 2)
                log.debug(f"  BPM (Essentia fallback): {bpm}")
        except Exception as exc:
            log.debug(f"  Essentia BPM failed: {exc}")

    # Key: Essentia always
    try:
        import essentia.standard as es
        audio = es.MonoLoader(filename=str(filepath), sampleRate=44100)()
        key, scale, _ = es.KeyExtractor()(audio)
        if key and scale:
            key_str = f"{key} {scale}"
            log.debug(f"  Key: {key_str}")
    except Exception as exc:
        log.debug(f"  Key extraction failed: {exc}")

    return bpm, key_str


# ── BPM/key subprocess wrapper ────────────────────────────────────────────────
# Bus errors (SIGBUS) from Essentia/librosa crash the whole process and can't
# be caught by try/except.  Running in a child process isolates the crash.

def _bpm_key_worker(filepath_str: str, result_file: str):
    """Child-process worker — writes result to a temp file, not a Queue.

    Using a temp file instead of multiprocessing.Queue avoids the POSIX
    semaphore corruption that causes malloc errors in the parent when this
    child crashes hard (SIGBUS, SIGABRT, etc.).
    """
    try:
        result = extract_bpm_key(Path(filepath_str))
    except Exception:
        result = (None, None)
    try:
        with open(result_file, "w") as f:
            json.dump(list(result), f)
    except Exception:
        pass


def safe_extract_bpm_key(filepath: Path,
                          timeout: int = 120) -> tuple[Optional[float], Optional[str]]:
    """Run BPM/key extraction in an isolated subprocess.

    IPC via temp file (not Queue) so a hard crash in the child can't corrupt
    the parent's heap through shared semaphores.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    result_file = tmp.name
    tmp.close()

    try:
        ctx = multiprocessing.get_context("spawn")
        p   = ctx.Process(
            target=_bpm_key_worker,
            args=(str(filepath), result_file),
            daemon=True,
        )
        p.start()
        p.join(timeout=timeout)

        if p.is_alive():
            log.warning("  BPM/key extraction timed out — skipping")
            p.terminate()
            p.join()
            return None, None

        if p.exitcode != 0:
            log.warning(f"  BPM/key extraction crashed (exit {p.exitcode}) — skipping")
            return None, None

        with open(result_file) as f:
            bpm, key = json.load(f)
        return bpm, key

    except Exception as exc:
        log.warning(f"  BPM/key subprocess error: {exc}")
        return None, None

    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass




# ── Album cover helpers ───────────────────────────────────────────────────────

def _make_cover_filename(artist: str, title: str) -> str:
    """Build a safe ASCII storage filename from artist + title.

    Strips diacritics (ä→a, ü→u, etc.) then removes anything that isn't
    alphanumeric, a hyphen, or underscore — same fix as the 'Invalid key' error
    caused by chars like ä in Supabase Storage paths.
    """
    raw = f"{artist}_{title}".lower()
    # Decompose unicode (ä → a + combining umlaut) then drop non-ASCII
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    raw = "_".join(filter(None, raw.split("_")))
    return raw[:150]


def _extract_album_art(filepath: Path) -> Optional[bytes]:
    """Pull embedded cover art bytes from an audio file, or None."""
    try:
        from mutagen import File
        from mutagen.mp3  import MP3
        from mutagen.flac import FLAC
        from mutagen.mp4  import MP4
        audio = File(str(filepath))
        if audio is None:
            return None
        if isinstance(audio, MP3):
            for tag in (audio.tags or {}).values():
                if hasattr(tag, "mime") and "image" in tag.mime:
                    return tag.data
        elif isinstance(audio, FLAC):
            if audio.pictures:
                return audio.pictures[0].data
        elif isinstance(audio, MP4):
            covers = (audio.tags or {}).get("covr")
            if covers:
                return bytes(covers[0])
    except Exception as exc:
        log.debug(f"  Cover art extraction failed for {filepath.name}: {exc}")
    return None


def _upload_cover(supabase, filepath: Path, artist: str, title: str) -> Optional[str]:
    """Extract cover art from file and upload to Supabase Storage.

    Returns the public URL string, or None if there's no art / upload fails.
    """
    image_data = _extract_album_art(filepath)
    if not image_data:
        return None
    filename   = _make_cover_filename(artist, title) + ".jpg"
    try:
        supabase.storage.from_(COVER_BUCKET).upload(
            filename,
            image_data,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        # Modern Supabase Python SDK returns the URL as a plain string
        url = supabase.storage.from_(COVER_BUCKET).get_public_url(filename)
        return url if isinstance(url, str) else url.get("publicUrl")
    except Exception as exc:
        log.warning(f"  Cover upload failed for {filepath.name}: {exc}")
        return None


# ── Audio storage upload ──────────────────────────────────────────────────────

def _upload_audio_file(supabase, filepath: Path, trackid: int) -> Optional[str]:
    """Upload a local audio file to Supabase Storage and return its public URL."""
    ext  = filepath.suffix.lower()
    mime = AUDIO_MIME_MAP.get(ext, "audio/mpeg")
    bucket_path = f"{trackid}{ext}"
    try:
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        supabase.storage.from_(AUDIO_BUCKET).upload(
            bucket_path,
            audio_bytes,
            file_options={"content-type": mime, "upsert": "true"},
        )
        url = supabase.storage.from_(AUDIO_BUCKET).get_public_url(bucket_path)
        if isinstance(url, dict):
            url = url.get("data", {}).get("publicUrl") or url.get("publicUrl")
        return url
    except Exception as exc:
        log.warning(f"  Audio upload failed for trackid {trackid}: {exc}")
        return None


# ── MFCC fingerprint (local file, no download needed) ─────────────────────────

def _compute_mfcc_fingerprint(filepath: Path) -> Optional[list]:
    """Compute a 13-element L2-normalised MFCC vector from a local audio file.

    Parameters match compute_audio_fingerprints.py and the browser JS implementation
    (HTK mel scale, same fmin/fmax/n_mels/n_mfcc).
    """
    try:
        import librosa
        y, sr = librosa.load(str(filepath), sr=FP_SR, mono=True,
                             duration=FP_DURATION, offset=FP_OFFSET)
        if len(y) < FP_SR * 5:
            y, sr = librosa.load(str(filepath), sr=FP_SR, mono=True,
                                 duration=FP_DURATION, offset=0.0)
        if len(y) < FP_SR * 2:
            log.warning("  MFCC: audio too short — skipping fingerprint")
            return None

        mfcc_frames = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=FP_N_MFCC,
            n_fft=FP_N_FFT,
            n_mels=FP_N_MELS,
            fmin=FP_FMIN,
            fmax=FP_FMAX,
            htk=True,
        )
        mfcc_mean  = np.mean(mfcc_frames, axis=1)
        mfcc_mean -= mfcc_mean.mean()
        norm = np.linalg.norm(mfcc_mean)
        if norm < 1e-10:
            log.warning("  MFCC: zero-norm vector (silent?) — skipping fingerprint")
            return None
        return (mfcc_mean / norm).tolist()
    except Exception as exc:
        log.warning(f"  MFCC fingerprint error: {exc}")
        return None


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_supabase():
    load_dotenv()
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


def load_existing_index(supabase) -> tuple[set, set, set]:
    """Load all tracks in one query and build three duplicate-detection sets.

    Returns:
        filepaths       — normalised filepaths already in DB
        title_artists   — (normalised_title, normalised_artist) tuples
        titles          — normalised titles (for same-title-different-artist check)
    """
    resp = supabase.table("tracks").select("filepath, title, artist").execute()
    filepaths, title_artists, titles = set(), set(), set()
    for t in (resp.data or []):
        fp     = (t.get("filepath") or "").strip().lower()
        title  = (t.get("title")   or "").strip().lower()
        artist = (t.get("artist")  or "").strip().lower()
        if fp:
            filepaths.add(fp)
        if title:
            title_artists.add((title, artist))
            titles.add(title)
    return filepaths, title_artists, titles


def upload_track(supabase, title: str, artist: str, filepath: str,
                 bpm: Optional[float], key: Optional[str],
                 embedding: list, album_art_url: Optional[str] = None) -> Optional[int]:
    try:
        track_data = {
            "title":         title,
            "artist":        artist,
            "filepath":      filepath,
            "embedding":     embedding,
            "bpm":           float(bpm) if bpm is not None else None,
            "key":           key,
            "album_art_url": album_art_url,
        }
        result = supabase.table("tracks").insert(track_data).execute()
        if not result.data:
            return None
        trackid = result.data[0]["trackid"]
        supabase.table("track_labels").insert({
            "trackid":       trackid,
            "semantic_tags": [],
            "energy":        None,
            "vibe":          [],
        }).execute()
        return trackid
    except Exception as exc:
        log.error(f"  Supabase upload error: {exc}")
        return None


# ── Post-processing helpers ───────────────────────────────────────────────────

def _run_script(script_rel_path: str, extra_args: list[str] = None):
    """Run a project script as a subprocess, inheriting the current env."""
    script = PROJECT_ROOT / script_rel_path
    cmd = [sys.executable, str(script)] + (extra_args or [])
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.warning(f"  Script exited with code {result.returncode}")


def _auto_tag_new_tracks(supabase, newly_added_ids: list[int]):
    """Inline ML auto-tagger — runs only on the newly ingested track IDs.

    Imports the core logic from backend/auto_tagger.py so there's no
    subprocess overhead and the same DB connection is reused.
    """
    if not newly_added_ids:
        return

    log.info(f"\nAuto-tagging {len(newly_added_ids)} new track(s) …")

    # Import shared logic from auto_tagger.py
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.auto_tagger import (
        load_all_tracks, load_existing_labels, build_embedding_matrix,
        get_suggestions, DEFAULT_MIN_SIMILARITY, MAX_GENRES, MAX_VIBES,
        SECONDARY_GENRE_RATIO,
    )

    all_tracks      = load_all_tracks(supabase)
    existing_labels = load_existing_labels(supabase)
    track_ids, X    = build_embedding_matrix(all_tracks)

    if not track_ids:
        log.warning("  No tracks with embeddings — skipping auto-tag.")
        return

    # Build label memory from trusted (human-tagged) tracks only
    label_memory, vibe_memory, energy_memory = {}, {}, {}
    for tid, row in existing_labels.items():
        if row.get("tag_source") not in {"manual", "auto_reviewed"}:
            continue
        tags   = row.get("semantic_tags") or []
        vibes  = row.get("vibe")          or []
        energy = row.get("energy")
        if isinstance(tags, list)  and tags:   label_memory[tid]  = tags
        if isinstance(vibes, list) and vibes:  vibe_memory[tid]   = vibes
        if energy is not None:                 energy_memory[tid] = float(energy)

    log.info(f"  {len(label_memory)} manually-tagged tracks as signal source")
    if len(label_memory) < 5:
        log.warning("  Very few manual tags — auto-tag quality may be low.")

    tagged  = 0
    skipped = 0

    for tid in newly_added_ids:
        genre_scores, vibe_scores, predicted_energy, max_sim = get_suggestions(
            tid, track_ids, X, label_memory, vibe_memory, energy_memory
        )

        if max_sim < DEFAULT_MIN_SIMILARITY or not genre_scores:
            log.info(f"  SKIP (sim={max_sim:.2f}, no signal) — trackid {tid}")
            skipped += 1
            continue

        # Cap genres (same rules as auto_tagger.py)
        final_tags = []
        top_tag, top_score = genre_scores[0]
        final_tags.append(top_tag)
        for tag, score in genre_scores[1:MAX_GENRES]:
            if score >= top_score * SECONDARY_GENRE_RATIO:
                final_tags.append(tag)

        final_vibes = [v for v, _ in vibe_scores[:MAX_VIBES]]
        energy      = round(predicted_energy) if predicted_energy is not None else None

        log.info(f"  TAG  (sim={max_sim:.2f}) trackid {tid} → "
                 f"tags={final_tags}  vibes={final_vibes}  energy={energy}")

        try:
            supabase.table("track_labels").update({
                "semantic_tags": final_tags,
                "vibe":          final_vibes,
                "energy":        energy,
                "tag_source":    "auto",
            }).eq("trackid", tid).execute()
            tagged += 1
        except Exception as exc:
            log.error(f"  Failed to write auto-tags for trackid {tid}: {exc}")

    log.info(f"  Auto-tag done — tagged: {tagged}  skipped: {skipped}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a music folder into DJMate (embed → tag → 3D)."
    )
    parser.add_argument("folder", help="Path to your mixing folder (scanned recursively)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Discover and report without writing anything to the DB.")
    parser.add_argument("--force-reembed", action="store_true",
                        help="Re-process tracks that are already in the DB.")
    parser.add_argument("--skip-auto-tag", action="store_true",
                        help="Skip running auto_tagger.py after ingestion.")
    parser.add_argument("--skip-3d",       action="store_true",
                        help="Skip re-running 3d_coordinator.py after ingestion.")
    parser.add_argument("--skip-covers",        action="store_true",
                        help="Skip uploading album cover art to Supabase Storage.")
    parser.add_argument("--skip-audio-upload",  action="store_true",
                        help="Skip uploading audio files to Supabase Storage.")
    parser.add_argument("--skip-fingerprints",  action="store_true",
                        help="Skip computing MFCC fingerprints for new tracks.")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        log.error(f"Folder not found: {folder}")
        sys.exit(1)

    log.info("=" * 65)
    log.info("DJMate — Music Ingestion Pipeline")
    log.info("=" * 65)
    log.info(f"Folder : {folder}")
    log.info(f"Dry run: {args.dry_run}")

    # ── 0. Load embedding model into isolated worker before touching the library ─
    # The worker subprocess runs MonoLoader + TF inference — if a bad file causes
    # a segfault or SIGBUS, only the child dies; the ingestor restarts it and
    # continues.  Warmup runs inside EmbeddingWorker._start().
    worker = None
    if not args.dry_run:
        model_path = _find_model()
        if model_path is None:
            log.error(
                "Essentia EffNet model not found. "
                "Set ESSENTIA_MODEL_PATH or place it at "
                f"{PROJECT_ROOT / 'models' / 'discogs-effnet-bs64.pb'}"
            )
            sys.exit(1)
        log.info(f"Loading embedding model from {model_path} …")
        try:
            worker = EmbeddingWorker(str(model_path))
        except RuntimeError as exc:
            log.error(f"Cannot continue: {exc}")
            sys.exit(1)

    # ── 1. Scan ───────────────────────────────────────────────────────────────
    audio_files = find_audio_files(folder)
    log.info(f"\nFound {len(audio_files)} audio files total")

    if not audio_files:
        log.warning("No audio files found. Nothing to do.")
        return

    # ── 2. Filter out already-ingested tracks ─────────────────────────────────
    if args.dry_run:
        to_process = audio_files
        log.info("[DRY RUN] Would attempt to process all found files.\n")
    else:
        supabase = get_supabase()
        if args.force_reembed:
            to_process = audio_files
        else:
            log.info("Checking database for existing tracks …")
            existing_fps, existing_ta, existing_titles = load_existing_index(supabase)
            to_process = []
            skipped_fp  = 0
            skipped_dup = 0
            for f in audio_files:
                # 1. Fast filepath check (no file read)
                if str(f).lower() in existing_fps:
                    skipped_fp += 1
                    continue
                # 2. Title / title+artist check (reads ID3 header only — fast)
                title, artist = read_tags(f)
                t_low = title.strip().lower()
                a_low = artist.strip().lower()
                if t_low and (
                    t_low in existing_titles
                    or (t_low, a_low) in existing_ta
                ):
                    log.info(f"  SKIP (duplicate title): {title!r} — {artist!r}")
                    skipped_dup += 1
                    continue
                to_process.append(f)
            log.info(f"  Already in DB (filepath) : {skipped_fp}")
            log.info(f"  Already in DB (title)    : {skipped_dup}")
            log.info(f"  New to ingest            : {len(to_process)}\n")

    if not to_process:
        log.info("✅  No new tracks to ingest.")
        if not args.skip_auto_tag:
            log.info("\nRunning auto-tagger on any untagged existing tracks …")
            _run_script("backend/auto_tagger.py")
        return

    # ── 4. Process each track ─────────────────────────────────────────────────
    stats = {"uploaded": 0, "failed": 0, "skipped": 0}
    newly_added_ids: list[int] = []

    for i, filepath in enumerate(to_process, 1):
        log.info(f"[{i}/{len(to_process)}] {filepath.name}")

        if args.dry_run:
            title, artist = read_tags(filepath)
            log.info(f"  DRY RUN — title={title!r}  artist={artist!r}")
            continue

        # Tags
        title, artist = read_tags(filepath)
        log.info(f"  title={title!r}  artist={artist!r}")

        # BPM + key (isolated subprocess — bus errors won't kill the ingestor)
        bpm, key = safe_extract_bpm_key(filepath)
        log.info(f"  bpm={bpm}  key={key}")

        # Embedding (runs in the persistent worker subprocess)
        emb_list = worker.compute(filepath)
        if emb_list is None:
            log.error(f"  ✗ Skipping — embedding failed")
            stats["failed"] += 1
            continue

        # Album cover
        cover_url = None
        if not args.skip_covers:
            cover_url = _upload_cover(supabase, filepath, artist, title)
            if cover_url:
                log.info(f"  ✓ Cover uploaded")
            else:
                log.debug(f"  No cover art found in file")

        # Upload track
        trackid = upload_track(
            supabase,
            title=title,
            artist=artist,
            filepath=str(filepath),
            bpm=bpm,
            key=key,
            embedding=emb_list,
            album_art_url=cover_url,
        )
        if trackid is not None:
            log.info(f"  ✓ Uploaded (trackid={trackid})")
            stats["uploaded"] += 1
            newly_added_ids.append(trackid)

            # Upload audio file to Supabase Storage
            if not args.skip_audio_upload:
                audio_url = _upload_audio_file(supabase, filepath, trackid)
                if audio_url:
                    supabase.table("tracks").update({"audio_url": audio_url}) \
                        .eq("trackid", trackid).execute()
                    log.info(f"  ✓ Audio uploaded to storage")

                    # Compute MFCC fingerprint from local file (no re-download needed)
                    if not args.skip_fingerprints:
                        fp = _compute_mfcc_fingerprint(filepath)
                        if fp:
                            supabase.table("track_features").upsert({
                                "trackid": trackid,
                                "mfcc":    fp,
                            }).execute()
                            log.info(f"  ✓ MFCC fingerprint stored")
                        else:
                            log.warning(f"  ✗ MFCC fingerprint failed — skipping")
                else:
                    log.warning(f"  ✗ Audio upload failed — fingerprint skipped")
        else:
            log.error(f"  ✗ Upload failed")
            stats["failed"] += 1

    # ── 5. Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("Ingestion complete")
    log.info(f"  Uploaded : {stats['uploaded']}")
    log.info(f"  Failed   : {stats['failed']}")
    if args.dry_run:
        log.info("  [DRY RUN — no changes written]")
    log.info("=" * 65)

    # Shut down the embedding worker — not needed for auto-tag or 3D steps
    if worker is not None:
        worker.shutdown()

    if args.dry_run or stats["uploaded"] == 0:
        return

    # ── 8. Auto-tag new tracks ────────────────────────────────────────────────
    if not args.skip_auto_tag:
        _auto_tag_new_tracks(supabase, newly_added_ids)
    else:
        log.info("\n[--skip-auto-tag] Skipping auto-tagger.")

    # ── 9. Rebuild 3-D UMAP coordinates ──────────────────────────────────────
    if not args.skip_3d:
        log.info("\nRebuilding 3-D UMAP map for frontend …")
        _run_script("scripts/3d_coordinator.py")
    else:
        log.info("\n[--skip-3d] Skipping 3d_coordinator.")

    log.info("\n✅  All done — new tracks are live in the frontend.")


if __name__ == "__main__":
    main()
