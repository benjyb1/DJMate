"""The local library index: one analysed entry per playable track.

Sources, merged by file path:

- rekordbox: every track in master.db whose file exists. rekordbox's own BPM,
  key, rating and artwork are trusted over anything we'd estimate.
- the Mixing folder: audio files rekordbox doesn't know about. These get BPM
  and key from Essentia during analysis.

Each entry carries the EffNet embedding and head scores from
:mod:`analyser`. On load (and after each build batch) the index derives the
arrays the suggestion engine scores against: a standardised embedding matrix
for similarity, and per-axis values (energy, darkness, vocals, electronic)
with library percentiles, so "darker" always means darker *relative to your
own library*.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import keys, paths
from .analyser import Analyser, AnalyserUnavailable
from .rekordbox import Rekordbox, RekordboxUnavailable

log = logging.getLogger(__name__)

INDEX_VERSION = 1
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
SAVE_EVERY = 20
BUILD_WORKERS = int(os.getenv("NEXT_TRACK_WORKERS", "3"))
MIN_TRACK_SECONDS = 60
# Clip identification. Calibrated on degraded 15 s "room" recordings of library
# tracks: the true track was top-1 in 8/10 and these thresholds kept the one
# wrong top-1 (margin 0.03) out.
ID_MIN_MATCH = 0.45
ID_MIN_MARGIN = 0.08
AXES = ("energy", "dark", "vocal", "electronic")


def track_id(path: str) -> str:
    return hashlib.sha1(path.lower().encode("utf-8")).hexdigest()[:12]


def _logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _is_bundled(path: str) -> bool:
    """rekordbox ships demo tracks and sampler one-shots in your library
    (~/Music/PioneerDJ). They're not records you'd mix into."""
    return "/PioneerDJ/" in path


_ARTIST_TITLE_RE = re.compile(r"^(.+?)\s+-\s+(.+)$")


def _read_tags(path: Path) -> tuple[str, str, str]:
    title = artist = album = None
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(str(path), easy=True)
        if audio is not None and audio.tags is not None:
            def first(k):
                v = audio.tags.get(k)
                return str(v[0]) if isinstance(v, list) and v else (str(v) if v else None)
            title, artist, album = first("title"), first("artist"), first("album")
    except Exception:
        pass
    if not title or not artist:
        m = _ARTIST_TITLE_RE.match(path.stem)
        if m:
            artist = artist or m.group(1).strip()
            title = title or m.group(2).strip()
    return title or path.stem, artist or "", album or ""


class LibraryIndex:
    def __init__(self, analyser: Optional[Analyser] = None,
                 rekordbox: Optional[Rekordbox] = None):
        self.analyser = analyser or Analyser()
        self.rekordbox = rekordbox or Rekordbox()
        self._lock = threading.RLock()
        self.entries: dict[str, dict] = {}
        self._emb: dict[str, np.ndarray] = {}
        self.externals: dict[str, dict] = {}
        self._ext_emb: dict[str, np.ndarray] = {}
        self.build_state = {"running": False, "done": 0, "total": 0,
                            "current": None, "started_at": None,
                            "finished_at": None, "error": None, "failed": 0}
        self._build_thread: Optional[threading.Thread] = None
        # derived, rebuilt by _derive()
        self.lib_ids: list[str] = []
        self.Xn = np.zeros((0, 0), dtype=np.float32)
        self.Xid = np.zeros((0, 0), dtype=np.float32)
        self._mu = None
        self._sd = None
        self.axis_vals: dict[str, np.ndarray] = {}
        self.axis_pct: dict[str, np.ndarray] = {}
        self._axis_sorted: dict[str, np.ndarray] = {}
        self._head_stats: dict[str, tuple[float, float]] = {}
        self._energy_model: Optional[dict] = None
        self._row_of: dict[str, int] = {}
        self._loaded_mtime: Optional[float] = None
        self.load()

    # ── persistence ─────────────────────────────────────────────────────────
    def _files(self):
        d = paths.data_dir()
        return d / "index.json", d / "embeddings.npy"

    def maybe_reload(self):
        """Pick up an index another process (the CLI build) has saved since."""
        if self.build_state["running"]:
            return
        meta_p, _ = self._files()
        try:
            mtime = meta_p.stat().st_mtime
        except OSError:
            return
        if mtime != self._loaded_mtime:
            self.load()

    def load(self):
        meta_p, emb_p = self._files()
        if not meta_p.exists():
            return
        try:
            self._loaded_mtime = meta_p.stat().st_mtime
            meta = json.loads(meta_p.read_text())
            if meta.get("version") != INDEX_VERSION:
                log.warning("Next Track index version changed; rebuilding from scratch")
                return
            mat = np.load(emb_p) if emb_p.exists() else np.zeros((0, 0))
            with self._lock:
                self.entries = {e["id"]: e for e in meta["entries"]}
                self._emb = {i: mat[r] for r, i in enumerate(meta.get("emb_ids", []))
                             if r < len(mat)}
                self._derive()
            log.info("Next Track index loaded: %d entries, %d analysed",
                     len(self.entries), len(self._emb))
        except Exception as exc:
            log.error("Could not load Next Track index: %s", exc)

    def save(self):
        meta_p, emb_p = self._files()
        with self._lock:
            ids = list(self._emb.keys())
            mat = (np.stack([self._emb[i] for i in ids]).astype(np.float32)
                   if ids else np.zeros((0, 0), dtype=np.float32))
            meta = {"version": INDEX_VERSION, "saved_at": time.time(),
                    "entries": list(self.entries.values()), "emb_ids": ids}
        tmp_meta = meta_p.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta))
        with open(emb_p.with_suffix(".tmp.npy"), "wb") as fh:
            np.save(fh, mat)
        os.replace(emb_p.with_suffix(".tmp.npy"), emb_p)
        os.replace(tmp_meta, meta_p)
        self._loaded_mtime = meta_p.stat().st_mtime

    # ── scanning ────────────────────────────────────────────────────────────
    def scan(self) -> list[dict]:
        """Collect every playable track from rekordbox and the Mixing folder."""
        found: dict[str, dict] = {}
        try:
            for row in self.rekordbox.library():
                if _is_bundled(row["path"]) or (row["duration"] or 999) < MIN_TRACK_SECONDS:
                    continue
                row["source"] = "rekordbox"
                found[row["path"].lower()] = row
        except RekordboxUnavailable as exc:
            log.warning("rekordbox unavailable during scan: %s", exc)

        root = paths.mixing_folder()
        if root.is_dir():
            for dirpath, _dirs, files in os.walk(root):
                for name in files:
                    p = Path(dirpath) / name
                    if p.suffix.lower() not in AUDIO_EXTENSIONS or name.startswith("._"):
                        continue
                    k = str(p).lower()
                    if k in found:
                        continue
                    title, artist, album = _read_tags(p)
                    found[k] = {"path": str(p), "title": title, "artist": artist,
                                "album": album, "bpm": None, "key_raw": None,
                                "rating": 0, "rb_genre": None, "duration": None,
                                "art_path": None, "rb_id": None, "source": "folder"}

        rows = []
        for row in found.values():
            p = Path(row["path"])
            try:
                st = p.stat()
            except OSError:
                continue
            row["id"] = track_id(row["path"])
            row["mtime"] = int(st.st_mtime)
            row["crate"] = (p.parent.name if str(p).startswith(str(root)) else None)
            row["camelot"] = keys.to_camelot(row.get("key_raw"))
            rows.append(row)
        return rows

    # ── building ────────────────────────────────────────────────────────────
    def start_build(self) -> bool:
        with self._lock:
            if self._build_thread and self._build_thread.is_alive():
                return False
            self._build_thread = threading.Thread(target=self._build, daemon=True,
                                                  name="next-track-index")
            self._build_thread.start()
            return True

    def _needs_analysis(self, row: dict) -> bool:
        old = self.entries.get(row["id"])
        if row["id"] not in self._emb or old is None:
            return True
        return old.get("mtime") != row["mtime"]

    def _build(self):
        st = self.build_state
        st.update(running=True, done=0, total=0, current="Scanning library",
                  started_at=time.time(), finished_at=None, error=None, failed=0)
        try:
            rows = self.scan()
            todo = []
            with self._lock:
                live = {r["id"] for r in rows}
                for r in rows:
                    old = self.entries.get(r["id"], {})
                    # Keep analysis results; refresh rekordbox metadata.
                    merged = {**old, **r}
                    if r["source"] == "folder":
                        # Folder tracks get BPM/key from analysis; don't wipe them.
                        for k in ("bpm", "key_raw", "camelot"):
                            if old.get(k) and not r.get(k):
                                merged[k] = old[k]
                    self.entries[r["id"]] = merged
                    if self._needs_analysis(merged):
                        todo.append(merged)
                for gone in set(self.entries) - live:
                    self.entries.pop(gone, None)
                    self._emb.pop(gone, None)
                self._derive()
            st["total"] = len(todo)
            log.info("Next Track build: %d tracks, %d to analyse", len(rows), len(todo))
            self._analyse_all(todo)
            with self._lock:
                self._derive()
                self.save()
        except AnalyserUnavailable as exc:
            st["error"] = str(exc)
            log.error("Next Track build stopped: %s", exc)
        except Exception as exc:
            st["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("Next Track build failed")
        finally:
            st.update(running=False, current=None, finished_at=time.time())

    def _analyse_all(self, todo: list[dict]):
        """Run the analysis queue across a few worker subprocesses. EffNet
        doesn't saturate an Apple Silicon CPU on its own, so two or three
        workers roughly halve a first full-library build."""
        if not todo:
            return
        st = self.build_state
        queue = list(reversed(todo))
        n_workers = max(1, min(BUILD_WORKERS, len(todo)))
        analysers = [self.analyser] + [Analyser() for _ in range(n_workers - 1)]
        fatal: list[BaseException] = []

        def worker(an: Analyser):
            while True:
                with self._lock:
                    if not queue or fatal:
                        return
                    e = queue.pop()
                    st["current"] = f"{e.get('artist') or ''} - {e.get('title') or ''}".strip(" -")
                want_rhythm = not e.get("bpm") or not e.get("camelot")
                try:
                    res = an.analyse(e["path"], want_rhythm=want_rhythm)
                except AnalyserUnavailable as exc:
                    with self._lock:
                        fatal.append(exc)
                    return
                with self._lock:
                    if res is None:
                        st["failed"] += 1
                        e["error"] = "analysis failed"
                    else:
                        self._apply_analysis(e, res)
                    st["done"] += 1
                    if st["done"] % SAVE_EVERY == 0:
                        self._derive()
                        self.save()

        threads = [threading.Thread(target=worker, args=(an,), daemon=True) for an in analysers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for an in analysers[1:]:
            an.shutdown()
        if fatal:
            raise fatal[0]

    def _apply_analysis(self, e: dict, res: dict):
        e["heads"] = res["heads"]
        e["genres"] = res["genres"]
        e.pop("error", None)
        if res.get("duration") and not e.get("duration"):
            e["duration"] = res["duration"]
        if not e.get("bpm") and res.get("bpm"):
            e["bpm"] = res["bpm"]
            e["bpm_estimated"] = True
        if not e.get("camelot") and res.get("key"):
            e["key_raw"] = res["key"]
            e["camelot"] = keys.to_camelot(res["key"])
            e["key_estimated"] = True
        self._emb[e["id"]] = np.asarray(res["emb"], dtype=np.float32)

    # ── derived arrays ──────────────────────────────────────────────────────
    def _raw_axes(self, heads_rows: list[dict], camelots: list, bpms: np.ndarray,
                  ratings: np.ndarray, fit: bool) -> dict[str, np.ndarray]:
        """Turn head probabilities into raw (unnormalised) axis values."""
        L = {h: _logit([r.get(h, 0.5) for r in heads_rows])
             for h in ("party", "aggressive", "relaxed", "electronic", "voice",
                       "happy", "sad")}
        if fit:
            self._head_stats = {h: (float(v.mean()), float(v.std() or 1.0))
                                for h, v in L.items()}
            self._bpm_stats = (float(np.nanmean(bpms)) if np.isfinite(bpms).any() else 125.0,
                               float(np.nanstd(bpms)) if np.isfinite(bpms).any() else 10.0)
        z = {h: (v - self._head_stats[h][0]) / self._head_stats[h][1] for h, v in L.items()}
        bpm_mu, bpm_sd = self._bpm_stats
        zb = np.nan_to_num((bpms - bpm_mu) / (bpm_sd or 1.0))

        minor = np.array([{True: 1.0, False: -1.0, None: 0.0}[keys.is_minor(c)]
                          for c in camelots])
        dark = -z["happy"] + 0.5 * z["sad"] + 0.5 * z["aggressive"] + 0.3 * minor

        F = np.column_stack([z["party"], z["aggressive"], z["relaxed"], zb])
        if fit:
            rated = ratings > 0
            if rated.sum() >= 25:
                A = np.column_stack([np.ones(rated.sum()), F[rated]])
                # Light ridge so a few hundred ratings can't overfit four features.
                lam = 1.0
                coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]),
                                       A.T @ ratings[rated])
                pred = np.column_stack([np.ones(len(F)), F]) @ coef
                corr = float(np.corrcoef(pred[rated], ratings[rated])[0, 1])
                self._energy_model = {"coef": coef.tolist(), "n": int(rated.sum()),
                                      "corr": round(corr, 3), "fitted": True}
            else:
                # Same blend as scripts/compute_effnet_energy.py, mapped to 1-5.
                self._energy_model = {"coef": [3.0, 0.45, 0.4, -0.3, 0.1],
                                      "n": int(rated.sum()), "fitted": False}
        coef = np.asarray(self._energy_model["coef"])
        pred = np.clip(np.column_stack([np.ones(len(F)), F]) @ coef, 1, 5)
        energy = np.where(ratings > 0, ratings + 0.25 * (pred - 3), pred)
        return {"energy": energy, "energy_pred": pred, "dark": dark,
                "vocal": L["voice"], "electronic": L["electronic"]}

    def _derive(self):
        ids = [i for i in self.entries
               if i in self._emb and self.entries[i].get("heads")
               and not _is_bundled(self.entries[i].get("path", ""))]
        self.lib_ids = ids
        self._row_of = {i: r for r, i in enumerate(ids)}
        if not ids:
            self.Xn = np.zeros((0, 0), dtype=np.float32)
            self.axis_vals, self.axis_pct = {}, {}
            return
        X = np.stack([self._emb[i] for i in ids]).astype(np.float32)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-6
        Z = (X - self._mu) / self._sd
        self.Xn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        # Identification of short clips (mic, snippets) works better on the
        # mean activations alone: a 15 s clip's frame-to-frame spread (std)
        # doesn't look like a whole track's.
        half = X.shape[1] // 2
        Zm = Z[:, :half]
        self.Xid = Zm / (np.linalg.norm(Zm, axis=1, keepdims=True) + 1e-9)

        ents = [self.entries[i] for i in ids]
        bpms = np.array([e.get("bpm") or np.nan for e in ents], dtype=np.float64)
        ratings = np.array([e.get("rating") or 0 for e in ents], dtype=np.float64)
        vals = self._raw_axes([e["heads"] for e in ents], [e.get("camelot") for e in ents],
                              bpms, ratings, fit=True)
        self.axis_vals = vals
        self.axis_pct, self._axis_sorted = {}, {}
        n = len(ids)
        for a in AXES:
            order = np.argsort(vals[a], kind="stable")
            pct = np.empty(n)
            pct[order] = np.arange(n) / max(n - 1, 1)
            self.axis_pct[a] = pct
            self._axis_sorted[a] = np.sort(vals[a])
        for r, e in enumerate(ents):
            e["energy"] = round(float(vals["energy"][r]), 2)

    # ── lookups used by the engine ──────────────────────────────────────────
    def features_for(self, tid: str) -> Optional[dict]:
        """Embedding (standardised, unit length), axis values and percentiles
        for a library track or a cached external one."""
        with self._lock:
            if tid in self._row_of:
                r = self._row_of[tid]
                return {"vec": self.Xn[r],
                        "axes": {a: float(self.axis_vals[a][r]) for a in AXES},
                        "pct": {a: float(self.axis_pct[a][r]) for a in AXES},
                        "row": r}
            if tid in self._ext_emb:
                return self._external_features(self.externals[tid], self._ext_emb[tid])
        return None

    def _external_features(self, e: dict, emb: np.ndarray) -> dict:
        z = (emb - self._mu) / self._sd
        vec = z / (np.linalg.norm(z) + 1e-9)
        bpm = np.array([e.get("bpm") or np.nan])
        vals = self._raw_axes([e["heads"]], [e.get("camelot")], bpm,
                              np.array([float(e.get("rating") or 0)]), fit=False)
        axes = {a: float(vals[a][0]) for a in AXES}
        pct = {a: float(np.searchsorted(self._axis_sorted[a], axes[a]) /
                        max(len(self._axis_sorted[a]) - 1, 1)) for a in AXES}
        e["energy"] = round(axes["energy"], 2)
        return {"vec": vec.astype(np.float32), "axes": axes,
                "pct": {a: min(1.0, v) for a, v in pct.items()}, "row": None}

    def identify(self, emb) -> Optional[dict]:
        """Which library track is this audio? Returns the best match, its
        score, and whether it's clear enough of the runner-up to trust."""
        with self._lock:
            if not self.lib_ids:
                return None
            emb = np.asarray(emb, dtype=np.float32)
            half = len(emb) // 2
            z = (emb[:half] - self._mu[:half]) / self._sd[:half]
            z /= np.linalg.norm(z) + 1e-9
            sims = self.Xid @ z
            order = np.argsort(-sims)[:2]
            top = float(sims[order[0]])
            second = float(sims[order[1]]) if len(order) > 1 else 0.0
            return {"id": self.lib_ids[int(order[0])], "match": round(top, 3),
                    "confident": top >= ID_MIN_MATCH and top - second >= ID_MIN_MARGIN}

    def get(self, tid: str) -> Optional[dict]:
        """An analysed version of the track if there is one: the library entry
        once it's been through the analyser, else an on-the-fly analysis."""
        with self._lock:
            if tid in self._row_of:
                return self.entries[tid]
            return self.externals.get(tid) or self.entries.get(tid)

    def is_analysed(self, tid: str) -> bool:
        with self._lock:
            return tid in self._row_of or tid in self.externals

    def find_by_path(self, path: str) -> Optional[dict]:
        return self.get(track_id(path))

    def add_external(self, meta: dict, res: dict) -> dict:
        """Register an analysed track that isn't in the library (a friend's
        USB, a mic recording, a dropped file) so it can be a starting point."""
        e = {**meta}
        e.setdefault("id", track_id(meta.get("path") or f"external:{time.time()}"))
        e["source"] = meta.get("source", "external")
        e["heads"] = res["heads"]
        e["genres"] = res["genres"]
        if not e.get("bpm") and res.get("bpm"):
            e["bpm"], e["bpm_estimated"] = res["bpm"], True
        if not e.get("camelot"):
            e["key_raw"] = e.get("key_raw") or res.get("key")
            e["camelot"] = keys.to_camelot(e["key_raw"])
            e["key_estimated"] = res.get("key") is not None
        e.setdefault("duration", res.get("duration"))
        with self._lock:
            self.externals[e["id"]] = e
            self._ext_emb[e["id"]] = np.asarray(res["emb"], dtype=np.float32)
            # Fill in its energy for display.
            if self.lib_ids:
                self._external_features(e, self._ext_emb[e["id"]])
        return e

    def status(self) -> dict:
        with self._lock:
            analysed = len(self.lib_ids)
            total = len(self.entries)
            sources = {}
            for e in self.entries.values():
                sources[e.get("source")] = sources.get(e.get("source"), 0) + 1
            return {"tracks": total, "analysed": analysed, "sources": sources,
                    "energy_model": self._energy_model,
                    "build": dict(self.build_state)}

    def search(self, q: str, limit: int = 12) -> list[dict]:
        terms = [t for t in q.lower().split() if t]
        if not terms:
            return []
        out, seen = [], set()
        with self._lock:
            for i in self.lib_ids:
                e = self.entries[i]
                hay = f"{e.get('title', '')} {e.get('artist', '')} {e.get('crate') or ''}".lower()
                # The same record often sits in several crate folders.
                name = f"{(e.get('artist') or '').lower()}|{(e.get('title') or '').lower()}"
                if all(t in hay for t in terms) and name not in seen:
                    seen.add(name)
                    out.append(e)
                    if len(out) >= limit:
                        break
        return out
