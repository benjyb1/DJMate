# backend/ingest_router.py — API endpoint to trigger the music ingestion pipeline
import asyncio
import subprocess
import sys
import os
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── In-memory state for the running ingest job ────────────────────────────────
_ingest_state = {
    "status": "idle",       # idle | running | done | error
    "folder": None,
    "log_lines": [],
    "process": None,
}


class IngestRequest(BaseModel):
    folder: str
    skip_auto_tag: bool = False
    skip_3d: bool = False
    skip_audio_upload: bool = False
    skip_fingerprints: bool = False


@router.post("/start")
async def start_ingest(req: IngestRequest, request: Request):
    """Kick off the ingest_music.py pipeline as a background subprocess."""
    if _ingest_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running")

    folder = Path(req.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    # Build command
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest_music.py"), str(folder)]
    if req.skip_auto_tag:
        cmd.append("--skip-auto-tag")
    if req.skip_3d:
        cmd.append("--skip-3d")
    if req.skip_audio_upload:
        cmd.append("--skip-audio-upload")
    if req.skip_fingerprints:
        cmd.append("--skip-fingerprints")

    # Extract tenant credentials from headers and pass to subprocess
    sb_url = request.headers.get("x-supabase-url", "")
    sb_key = request.headers.get("x-supabase-key", "")
    env = os.environ.copy()
    if sb_url and sb_key:
        env["SUPABASE_URL"] = sb_url
        env["SUPABASE_KEY"] = sb_key

    # Reset state
    _ingest_state["status"] = "running"
    _ingest_state["folder"] = str(folder)
    _ingest_state["log_lines"] = []
    _ingest_state["process"] = None

    # Launch in background
    asyncio.get_event_loop().create_task(_run_ingest(cmd, env=env))

    return {"status": "started", "folder": str(folder)}


async def _run_ingest(cmd: list[str], env: dict | None = None):
    """Run the ingest subprocess and capture output line-by-line."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        _ingest_state["process"] = proc

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            _ingest_state["log_lines"].append(line)
            # Keep a rolling window so memory doesn't blow up
            if len(_ingest_state["log_lines"]) > 500:
                _ingest_state["log_lines"] = _ingest_state["log_lines"][-300:]

        await proc.wait()
        _ingest_state["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        _ingest_state["log_lines"].append(f"ERROR: {exc}")
        _ingest_state["status"] = "error"
    finally:
        _ingest_state["process"] = None


@router.get("/status")
async def ingest_status(since: int = 0):
    """
    Poll the current ingest job status.
    Pass `since=N` to get only log lines after index N (for incremental updates).
    """
    lines = _ingest_state["log_lines"]
    return {
        "status": _ingest_state["status"],
        "folder": _ingest_state["folder"],
        "log_lines": lines[since:],
        "total_lines": len(lines),
    }
