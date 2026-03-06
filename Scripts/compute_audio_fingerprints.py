#!/usr/bin/env python3
"""
Compute MFCC audio fingerprints for every track in the library and store
them in the track_features table.

Algorithm (consistent with the browser-side implementation in LiveMode.jsx):
  • Load 90 s of audio starting at 15 s (skips intros, captures the body)
  • Compute 13 MFCC coefficients using a 40-band mel filterbank (HTK scale,
    fmin=20 Hz, fmax=8000 Hz, n_fft=2048, sr=22050)
  • Mean-centre the coefficient vector (removes recording-level bias)
  • L2-normalise (unit vector — cosine similarity in the browser)

Usage:
    pip install librosa supabase python-dotenv requests
    python scripts/compute_audio_fingerprints.py
"""

import os
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import librosa
import requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SR        = 22050    # analysis sample rate (librosa default)
N_FFT     = 2048     # FFT window size
N_MELS    = 40       # mel filterbank bands (matches JS implementation)
N_MFCC    = 13       # number of MFCC coefficients to keep
FMIN      = 20.0     # mel filterbank lower bound (Hz)
FMAX      = 8000.0   # mel filterbank upper bound (Hz)
OFFSET    = 15.0     # skip first 15 s (typical DJ intro)
DURATION  = 90.0     # analyse 90 s of the track body

PROGRESS_FILE = Path(__file__).parent / "fingerprint_progress.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": [], "failed": []}


def save_progress(prog):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)


def download_audio(url: str, dest: str) -> bool:
    """Stream-download an audio file.  Returns True on success."""
    try:
        with requests.get(url, timeout=90, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 17):   # 128 KB chunks
                    f.write(chunk)
        return True
    except Exception as exc:
        print(f"    ✗ download failed: {exc}")
        return False


def compute_mfcc_fingerprint(path: str) -> list[float] | None:
    """
    Returns a 13-element L2-normalised MFCC vector, or None on failure.

    Parameters are chosen to match the mel filterbank built on the browser
    side (HTK mel scale, same fmin/fmax, same n_mels/n_mfcc).
    """
    try:
        # Try loading from OFFSET first; fall back to start if the file is short
        y, sr = librosa.load(path, sr=SR, mono=True,
                             duration=DURATION, offset=OFFSET)
        if len(y) < SR * 5:
            y, sr = librosa.load(path, sr=SR, mono=True,
                                 duration=DURATION, offset=0.0)
        if len(y) < SR * 2:
            print("    ✗ audio too short")
            return None

        # MFCC — using htk=True so the mel scale matches the browser JS formula
        mfcc_frames = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            htk=True,
        )                                     # shape: (13, T)

        mfcc_mean = np.mean(mfcc_frames, axis=1)   # (13,)  time-averaged
        mfcc_mean -= mfcc_mean.mean()              # mean-centre (removes DC bias)
        norm = np.linalg.norm(mfcc_mean)
        if norm < 1e-10:
            print("    ✗ zero-norm vector (silent track?)")
            return None
        mfcc_norm = mfcc_mean / norm               # unit vector

        return mfcc_norm.tolist()

    except Exception as exc:
        print(f"    ✗ fingerprint error: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("SUPABASE_URL / SUPABASE_KEY not set in .env")

    sb   = create_client(SUPABASE_URL, SUPABASE_KEY)
    prog = load_progress()
    done_ids   = set(map(str, prog["done"]))
    failed_ids = set(map(str, prog["failed"]))

    # Fetch all tracks that have an audio URL
    rows = sb.table("tracks").select("trackid, title, artist, audio_url").execute().data
    tracks_with_audio = [t for t in rows if t.get("audio_url")]
    remaining = [t for t in tracks_with_audio
                 if str(t["trackid"]) not in done_ids]

    print(f"Tracks with audio : {len(tracks_with_audio)}")
    print(f"Already done      : {len(done_ids)}")
    print(f"Remaining         : {len(remaining)}\n")

    newly_done   = 0
    newly_failed = 0
    t0 = time.time()

    for idx, track in enumerate(remaining):
        tid    = str(track["trackid"])
        title  = track.get("title",  "?")
        artist = track.get("artist", "?")
        url    = track["audio_url"]

        suffix = Path(url).suffix or ".mp3"
        print(f"[{idx+1}/{len(remaining)}] {artist} — {title}")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if not download_audio(url, tmp_path):
                failed_ids.add(tid)
                newly_failed += 1
                continue

            fp = compute_mfcc_fingerprint(tmp_path)
            if fp is None:
                failed_ids.add(tid)
                newly_failed += 1
                continue

            # Upsert into track_features
            sb.table("track_features").upsert({
                "trackid": int(tid),
                "mfcc":    fp,
            }).execute()

            done_ids.add(tid)
            newly_done += 1
            elapsed = time.time() - t0
            rate    = newly_done / elapsed if elapsed > 0 else 0
            eta_s   = (len(remaining) - (idx + 1)) / rate if rate > 0 else 0
            print(f"    ✓  [{newly_done} done | "
                  f"ETA {int(eta_s//60)}m{int(eta_s%60):02d}s]")

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Save progress every 20 tracks
        if (idx + 1) % 20 == 0:
            save_progress({"done": list(done_ids), "failed": list(failed_ids)})
            print(f"  ── progress saved ──\n")

    save_progress({"done": list(done_ids), "failed": list(failed_ids)})
    elapsed = time.time() - t0
    print(f"\nFinished in {int(elapsed//60)}m{int(elapsed%60):02d}s")
    print(f"Done: {len(done_ids)}   Failed: {len(failed_ids)}")
    if failed_ids:
        print(f"Failed IDs: {sorted(failed_ids)[:20]}")


if __name__ == "__main__":
    main()
