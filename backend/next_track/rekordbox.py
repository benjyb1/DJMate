"""Read-only access to the local rekordbox 6/7 database.

Two jobs:

- list the library (title, artist, BPM, key, rating, artwork) for every track
  whose file actually exists on this Mac,
- report what's playing, from rekordbox's play history. rekordbox appends a
  row to ``djmdSongHistory`` as each track is played, so the newest row is the
  track on air and the rest of that history session is what's been played.

pyrekordbox opens the encrypted master.db without touching it. Reading is safe
while rekordbox is running; we never write.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class RekordboxUnavailable(RuntimeError):
    pass


class Rekordbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._db = None
        self._error: Optional[str] = None

    # ── connection ──────────────────────────────────────────────────────────
    def _open(self):
        if self._db is not None:
            return self._db
        try:
            from pyrekordbox import Rekordbox6Database
        except ImportError as exc:
            self._error = "pyrekordbox is not installed"
            raise RekordboxUnavailable(self._error) from exc
        try:
            self._db = Rekordbox6Database()
            self._error = None
        except Exception as exc:
            self._error = f"Could not open the rekordbox database: {exc}"
            raise RekordboxUnavailable(self._error) from exc
        return self._db

    def available(self) -> tuple[bool, Optional[str]]:
        with self._lock:
            try:
                self._open()
                return True, None
            except RekordboxUnavailable as exc:
                return False, str(exc)

    def _fresh(self):
        """Return the db with any stale read transaction ended, so rows
        rekordbox wrote since our last query become visible."""
        db = self._open()
        try:
            db.session.rollback()
        except Exception:
            # The connection went bad (rekordbox restarted, file replaced).
            self._db = None
            db = self._open()
        return db

    # ── library ─────────────────────────────────────────────────────────────
    @staticmethod
    def _row(c, share_dir: Path) -> dict:
        key = c.Key.ScaleName if c.Key else None
        art = None
        if c.ImagePath:
            p = share_dir / c.ImagePath.lstrip("/")
            if p.is_file():
                art = str(p)
        return {
            "rb_id": str(c.ID),
            "path": c.FolderPath,
            "title": c.Title or Path(c.FolderPath or "").stem,
            "artist": c.Artist.Name if c.Artist else "",
            "album": c.Album.Name if c.Album else "",
            "bpm": round(c.BPM / 100.0, 2) if c.BPM else None,
            "key_raw": key,
            "rating": int(c.Rating or 0),
            "rb_genre": c.Genre.Name if c.Genre else None,
            "duration": float(c.Length) if c.Length else None,
            "art_path": art,
        }

    def library(self) -> list[dict]:
        """Every rekordbox track whose audio file exists locally."""
        with self._lock:
            db = self._fresh()
            share = Path(db.share_directory)
            out = []
            for c in db.get_content().all():
                fp = c.FolderPath
                if not fp or not os.path.isfile(fp):
                    continue
                try:
                    out.append(self._row(c, share))
                except Exception as exc:  # one odd row shouldn't sink the scan
                    log.debug("Skipping rekordbox row %s: %s", c.ID, exc)
            return out

    # ── now playing ─────────────────────────────────────────────────────────
    def now_playing(self) -> Optional[dict]:
        """Newest play-history row plus the rest of its session.

        Returns None when rekordbox has never logged a play.
        """
        from pyrekordbox.db6 import tables as t

        with self._lock:
            db = self._fresh()
            share = Path(db.share_directory)
            latest = (db.query(t.DjmdSongHistory)
                      .order_by(t.DjmdSongHistory.created_at.desc())
                      .first())
            if latest is None:
                return None
            session_rows = (db.query(t.DjmdSongHistory)
                            .filter(t.DjmdSongHistory.HistoryID == latest.HistoryID)
                            .order_by(t.DjmdSongHistory.created_at.asc())
                            .all())
            played = []
            for r in session_rows:
                c = r.Content
                if c is None:
                    continue
                row = self._row(c, share)
                row["played_at"] = r.created_at.isoformat() if r.created_at else None
                played.append(row)
            if not played:
                return None
            current = played[-1]
            started = latest.created_at
            return {
                "current": current,
                "played": played,
                "history_id": str(latest.HistoryID),
                "started_at": started.isoformat() if started else None,
                # rekordbox stores local wall-clock time without a zone.
                "age_seconds": (datetime.now() - started).total_seconds() if started else None,
            }
