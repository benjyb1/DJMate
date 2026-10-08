"""HTTP API for the Next Track screen (mounted at /next).

Local-only by design: rekordbox, the audio files and the Essentia models all
live on this Mac. On a server without them (Render) every endpoint reports
itself unavailable instead of crashing the app.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from . import engine, paths
from .index import LibraryIndex, track_id
from .rekordbox import RekordboxUnavailable

log = logging.getLogger(__name__)
router = APIRouter()

_state_lock = threading.Lock()
_index: Optional[LibraryIndex] = None
_pending: dict[str, dict] = {}          # id -> {"started": t, "error": str|None}


def _idx() -> LibraryIndex:
    global _index
    with _state_lock:
        if _index is None:
            _index = LibraryIndex()
        else:
            _index.maybe_reload()
        return _index


def _require_index() -> LibraryIndex:
    idx = _idx()
    if not idx.lib_ids:
        raise HTTPException(409, "The library hasn't been analysed yet. Run a scan first.")
    return idx


# ── status & indexing ───────────────────────────────────────────────────────
@router.get("/status")
def status():
    idx = _idx()
    rb_ok, rb_err = idx.rekordbox.available()
    return {
        "rekordbox": {"available": rb_ok, "error": rb_err},
        "analyser": {"model_found": paths.effnet_model() is not None},
        "index": idx.status(),
        "directions": [{"id": d.id, "label": d.label} for d in engine.DIRECTIONS.values()],
        "default_directions": engine.DEFAULT_DIRECTIONS,
    }


@router.post("/index/build")
def build_index():
    started = _idx().start_build()
    return {"started": started, "build": _idx().build_state}


# ── now playing ─────────────────────────────────────────────────────────────
def _analyse_external_in_background(row: dict):
    tid = track_id(row["path"])
    with _state_lock:
        if tid in _pending and _pending[tid].get("error") is None:
            return
        _pending[tid] = {"started": time.time(), "error": None}

    def work():
        idx = _idx()
        res = idx.analyser.analyse(row["path"], want_rhythm=not row.get("bpm"))
        with _state_lock:
            if res is None:
                _pending[tid]["error"] = "Couldn't analyse this file"
                return
            idx.add_external({**row, "id": tid, "source": "external"}, res)
            _pending.pop(tid, None)

    threading.Thread(target=work, daemon=True, name=f"analyse-{tid}").start()


@router.get("/now")
def now_playing():
    """What rekordbox says is on air, and what's been played this session."""
    idx = _idx()
    try:
        np_ = idx.rekordbox.now_playing()
    except RekordboxUnavailable as exc:
        return {"available": False, "error": str(exc)}
    if np_ is None:
        return {"available": True, "track": None, "played_ids": []}

    cur = np_["current"]
    tid = track_id(cur["path"]) if cur.get("path") else None
    entry = idx.get(tid) if tid else None
    state = "ready"
    if entry is None or not idx.is_analysed(tid):
        # Not in the analysed library: a friend's USB, a new download.
        if cur.get("path") and os.path.isfile(cur["path"]):
            pending = _pending.get(tid)
            if pending and pending.get("error"):
                state = "failed"
            else:
                state = "analysing"
                _analyse_external_in_background(cur)
        else:
            state = "missing_file"
    track = engine.public_track(entry) if entry is not None and state == "ready" else {
        "id": tid, "title": cur.get("title"), "artist": cur.get("artist"),
        "bpm": cur.get("bpm"), "key": None, "source": "external",
    }
    return {
        "available": True,
        "state": state,
        "track": track,
        "played_ids": [track_id(p["path"]) for p in np_["played"] if p.get("path")],
        "played": [{"id": track_id(p["path"]), "title": p.get("title"),
                    "artist": p.get("artist"), "played_at": p.get("played_at")}
                   for p in np_["played"] if p.get("path")],
        "history_id": np_["history_id"],
        "started_at": np_["started_at"],
        "age_seconds": np_["age_seconds"],
    }


# ── suggestions ─────────────────────────────────────────────────────────────
@router.get("/suggest")
def suggest(id: str, exclude: str = "", directions: str = "",
            alternates: int = Query(2, ge=0, le=5)):
    idx = _require_index()
    ex = {x for x in exclude.split(",") if x}
    dirs = [d for d in directions.split(",") if d] or None
    try:
        return engine.suggest(idx, id, exclude_ids=ex, directions=dirs,
                              alternates=alternates)
    except KeyError:
        raise HTTPException(404, "Track not found in the analysed library")


@router.get("/track/{tid}")
def get_track(tid: str):
    e = _idx().get(tid)
    if e is None:
        raise HTTPException(404, "Unknown track")
    return engine.public_track(e)


@router.get("/search")
def search(q: str, limit: int = Query(12, ge=1, le=50)):
    return [engine.public_track(e) for e in _require_index().search(q, limit)]


# ── analysing audio that isn't in the library ───────────────────────────────
@router.post("/analyse")
async def analyse_upload(file: UploadFile = File(...), kind: str = Form("file")):
    """Analyse a dropped file or a microphone capture, find where it sits in
    your library, and register it as a starting point for suggestions."""
    idx = _require_index()
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        res = await _run_blocking(idx.analyser.analyse, tmp.name, True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if res is None:
        raise HTTPException(422, "Couldn't analyse that audio. Is it long enough?")

    if kind == "mic":
        title, artist = f"Heard at {datetime.now():%H:%M}", "Microphone"
    else:
        stem = Path(file.filename or "Dropped file").stem
        artist, _, title = stem.partition(" - ")
        title, artist = (title or stem), (artist if title else "")
    e = idx.add_external({"id": f"x{int(time.time() * 1000):x}", "title": title,
                          "artist": artist, "source": kind}, res)

    # Is it something you already own? If we're confident, the frontend centres
    # on your copy (with rekordbox's exact BPM and key) instead of the clip.
    nearest = None
    hit = idx.identify(res["emb"])
    if hit is not None:
        nearest = {"track": engine.public_track(idx.get(hit["id"])),
                   "match": hit["match"], "confident": hit["confident"]}
    return {"track": engine.public_track(e), "nearest": nearest}


async def _run_blocking(fn, *args):
    import anyio
    return await anyio.to_thread.run_sync(lambda: fn(*args))


# ── artwork & Finder ────────────────────────────────────────────────────────
def _embedded_art(path: str) -> Optional[bytes]:
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(path)
        if audio is None:
            return None
        pics = getattr(audio, "pictures", None)       # FLAC
        if pics:
            return pics[0].data
        tags = audio.tags or {}
        for k in list(tags.keys()):
            if k.startswith("APIC"):                    # MP3 / AIFF ID3
                return tags[k].data
        covr = tags.get("covr") if hasattr(tags, "get") else None   # M4A
        if covr:
            return bytes(covr[0])
    except Exception:
        return None
    return None


@router.get("/art/{tid}")
def artwork(tid: str):
    e = _idx().get(tid)
    if e is None:
        raise HTTPException(404)
    if e.get("art_path") and os.path.isfile(e["art_path"]):
        return FileResponse(e["art_path"], headers={"Cache-Control": "max-age=86400"})
    cache = paths.art_cache_dir() / f"{tid}.img"
    miss = paths.art_cache_dir() / f"{tid}.none"
    if cache.exists():
        data = cache.read_bytes()
    elif miss.exists() or not e.get("path"):
        raise HTTPException(404)
    else:
        data = _embedded_art(e["path"])
        if not data:
            miss.touch()
            raise HTTPException(404)
        cache.write_bytes(data)
    mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return Response(data, media_type=mime, headers={"Cache-Control": "max-age=86400"})


@router.post("/reveal/{tid}")
def reveal(tid: str):
    """Show the file in Finder, ready to drag onto a rekordbox deck."""
    e = _idx().get(tid)
    if e is None or not e.get("path") or not os.path.isfile(e["path"]):
        raise HTTPException(404, "File not found")
    subprocess.Popen(["open", "-R", e["path"]])
    return {"ok": True}
