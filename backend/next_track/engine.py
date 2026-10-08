"""Directional suggestions: from the track on air, where can you go next?

Every direction answers the same question with a different push:

    "Which mixable track is closest to this one, *but* noticeably more X?"

"Mixable" is a hard filter first: tempo within reach of the pitch fader, not
the track that's playing, not anything already played this session. Inside
that pool each direction scores candidates on:

- similarity: cosine distance between standardised EffNet embeddings, ranked
  within the pool so the scale is comparable across tracks,
- shift: how far the candidate moves along the direction's axis, measured in
  library percentiles (so "darker" means darker than this track relative to
  *your* music, not to some global scale), saturating so the pick is a step,
  not a leap,
- key: Camelot-wheel compatibility,
- tempo: closeness in BPM.

Directions are filled in priority order and a track is used at most once, so
the screen never shows the same record twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import keys
from .index import LibraryIndex


@dataclass(frozen=True)
class Direction:
    id: str
    label: str
    axis: str            # 'energy' | 'dark' | 'vocal' | 'electronic' | 'bpm' | 'similar'
    sign: int = 1


DIRECTIONS = {
    d.id: d for d in [
        Direction("closest", "Closest blend", "similar"),
        Direction("energy_up", "More energy", "energy", +1),
        Direction("energy_down", "Bring it down", "energy", -1),
        Direction("faster", "Faster", "bpm", +1),
        Direction("slower", "Slower", "bpm", -1),
        Direction("darker", "Darker", "dark", +1),
        Direction("lighter", "Lighter", "dark", -1),
        Direction("vocal", "More vocal", "vocal", +1),
        Direction("instrumental", "Instrumental", "vocal", -1),
        Direction("electronic", "More electronic", "electronic", +1),
        Direction("organic", "More organic", "electronic", -1),
    ]
}

DEFAULT_DIRECTIONS = ["energy_up", "darker", "faster", "electronic",
                      "energy_down", "slower", "vocal", "closest"]

# Assignment order: the most distinctive pushes pick first, "closest" last so
# it doesn't steal the obvious neighbour from a direction that needed it.
_PRIORITY = ["energy_up", "energy_down", "faster", "slower", "darker", "lighter",
             "vocal", "instrumental", "electronic", "organic", "closest"]

MIN_SHIFT = 0.10        # percentile points along the axis
FULL_SHIFT = 0.35       # shift beyond this earns no extra credit
BPM_WINDOW = 0.06       # +/- 6% for non-tempo directions
RELAXED_MIN_SHIFT = 0.03
RELAXED_BPM_WINDOW = 0.09
BPM_STEP = (1.5, 10.0)  # faster/slower must move this many BPM
BPM_STEP_PCT = 0.09     # ...and no more than 9%


def _bpm_closeness(bpm_c: np.ndarray, bpm0: Optional[float]) -> np.ndarray:
    if not bpm0:
        return np.full(len(bpm_c), 0.5)
    d = np.abs(np.nan_to_num(bpm_c, nan=bpm0 * 1.5) - bpm0)
    return np.clip(1 - d / 8.0, 0, 1)


def _rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.ones(len(x))
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x)) / (len(x) - 1)
    return r


def _norm_title(e: dict) -> str:
    t = (e.get("title") or "").lower()
    for junk in ("(original mix)", "(extended mix)", "original mix", "extended mix"):
        t = t.replace(junk, "")
    return f"{(e.get('artist') or '').lower().strip()}|{t.strip()}"


def suggest(index: LibraryIndex, current_id: str,
            exclude_ids: Optional[set[str]] = None,
            directions: Optional[list[str]] = None,
            alternates: int = 2) -> dict:
    """Suggestions for every requested direction from ``current_id``."""
    directions = [d for d in (directions or DEFAULT_DIRECTIONS) if d in DIRECTIONS]
    feats = index.features_for(current_id)
    cur = index.get(current_id)
    if feats is None or cur is None:
        raise KeyError(current_id)

    with index._lock:
        ids = list(index.lib_ids)
        ents = [index.entries[i] for i in ids]
        Xn = index.Xn
        pct = {a: index.axis_pct[a] for a in index.axis_pct}

    exclude = set(exclude_ids or ())
    exclude.add(current_id)
    cur_name = _norm_title(cur)
    exclude_names = {cur_name}
    for x in exclude:
        e = index.get(x)
        if e:
            exclude_names.add(_norm_title(e))

    bpm0 = cur.get("bpm")
    cam0 = cur.get("camelot")
    bpm = np.array([e.get("bpm") or np.nan for e in ents], dtype=np.float64)
    key_score = np.array([keys.compatibility(e.get("camelot"), cam0) for e in ents])
    sims = Xn @ feats["vec"] if len(ids) else np.zeros(0)

    pool = np.array([(i not in exclude) and (_norm_title(e) not in exclude_names)
                     for i, e in zip(ids, ents)], dtype=bool)
    if bpm0:
        rel = (bpm - bpm0) / bpm0
        steady = np.abs(np.nan_to_num(rel, nan=1.0)) <= BPM_WINDOW
    else:
        rel = np.zeros(len(ids))
        steady = np.ones(len(ids), dtype=bool)

    sim_rank = np.zeros(len(ids))
    if pool.any():
        sim_rank[pool] = _rank01(sims[pool])
    bpm_close = _bpm_closeness(bpm, bpm0)

    used: set[str] = set()
    used_names: set[str] = set()
    results: dict[str, dict] = {}

    for did in sorted(directions, key=_PRIORITY.index):
        d = DIRECTIONS[did]
        base = pool.copy()
        if d.axis == "bpm":
            if not bpm0:
                results[did] = {"id": did, "label": d.label, "tracks": [],
                                "empty_reason": "No BPM for the current track"}
                continue
            step = d.sign * (bpm - bpm0)
            ok = (step >= BPM_STEP[0]) & (step <= BPM_STEP[1]) & \
                 (np.abs(np.nan_to_num(rel, nan=1.0)) <= BPM_STEP_PCT)
            mask = base & ok
            shift = np.clip(1 - np.abs(np.nan_to_num(step) - 4.0) / 6.0, 0, 1)
            score = 0.50 * sim_rank + 0.25 * shift + 0.25 * key_score
        elif d.axis == "similar":
            mask = base & steady
            if mask.sum() < 1 + alternates and bpm0:
                mask = base & (np.abs(np.nan_to_num(rel, nan=1.0)) <= RELAXED_BPM_WINDOW)
            score = 0.65 * sim_rank + 0.25 * key_score + 0.10 * bpm_close
            shift = np.zeros(len(ids))
        else:
            delta = d.sign * (pct[d.axis] - feats["pct"][d.axis])
            mask = base & steady & (delta >= MIN_SHIFT)
            if mask.sum() < 1 + alternates:
                # The track already sits near the end of this axis (the darkest
                # record in the crate can't get much darker). Accept a smaller
                # push and a little more tempo room rather than show nothing.
                wide = np.abs(np.nan_to_num(rel, nan=1.0)) <= RELAXED_BPM_WINDOW if bpm0 else steady
                mask = base & wide & (delta >= RELAXED_MIN_SHIFT)
            shift = np.clip(delta / FULL_SHIFT, 0, 1)
            score = (0.45 * sim_rank + 0.30 * shift + 0.15 * key_score
                     + 0.10 * bpm_close)

        # Prefer harmonically safe picks; fall back to anything in range.
        strict = mask & (key_score >= 0.5)
        chosen_mask = strict if strict.sum() >= 1 + alternates else mask
        order = np.argsort(-np.where(chosen_mask, score, -np.inf))
        picks = []
        for r in order:
            if not chosen_mask[r]:
                break
            tid = ids[r]
            name = _norm_title(ents[r])
            if tid in used or name in used_names:
                continue
            picks.append(r)
            used_names.add(name)
            if len(picks) >= 1 + alternates:
                break
        if picks:
            used.add(ids[picks[0]])

        tracks = []
        for r in picks:
            e = ents[r]
            tracks.append({
                **public_track(e),
                "match": round(float(sims[r]), 3),
                "score": round(float(score[r]), 3),
                "key_compat": round(float(key_score[r]), 2),
                "shift": (round(float(d.sign * (pct[d.axis][r] - feats["pct"][d.axis])), 3)
                          if d.axis in pct else None),
                "reasons": _reasons(d, e, cur, feats, pct, r),
            })
        results[did] = {"id": did, "label": d.label, "tracks": tracks,
                        "empty_reason": None if tracks else _empty_reason(d, feats)}

    return {
        "current": {**public_track(cur), "pct": feats["pct"]},
        "directions": [results[d] for d in directions if d in results],
        "pool_size": int(pool.sum()),
    }


_EXTREMES = {
    ("energy", 1): "Already one of your most energetic",
    ("energy", -1): "Already one of your calmest",
    ("dark", 1): "Already one of your darkest",
    ("dark", -1): "Already one of your lightest",
    ("vocal", 1): "Already one of your most vocal",
    ("vocal", -1): "Already about as instrumental as it gets",
    ("electronic", 1): "Already one of your most electronic",
    ("electronic", -1): "Already one of your most organic",
}


def _empty_reason(d: Direction, feats: dict) -> str:
    if d.axis in feats["pct"]:
        p = feats["pct"][d.axis]
        if (d.sign > 0 and p >= 0.85) or (d.sign < 0 and p <= 0.15):
            return _EXTREMES[(d.axis, d.sign)]
    if d.axis == "bpm":
        return "Nothing in key at that tempo"
    return "Nothing mixable that way yet"


def _reasons(d: Direction, e: dict, cur: dict, feats: dict, pct: dict, r: int) -> list[str]:
    """Short chips explaining a pick, most relevant to the direction first."""
    words = {"energy": ("energy", "calmer"), "dark": ("darker", "lighter"),
             "vocal": ("vocal", "less vocal"), "electronic": ("electronic", "organic")}
    axis_chip = None
    if d.axis in words:
        delta = pct[d.axis][r] - feats["pct"][d.axis]
        axis_chip = f"{words[d.axis][0 if delta > 0 else 1]} +{round(abs(delta) * 100)}"

    bpm_chip = None
    b0, b1 = cur.get("bpm"), e.get("bpm")
    if b1:
        diff = (b1 - b0) if b0 else 0
        if abs(diff) < 0.5:
            bpm_chip = f"{b1:.0f} BPM"
        else:
            mag = f"{abs(diff):.0f}" if abs(diff) >= 1.5 else f"{abs(diff):.1f}"
            bpm_chip = f"{'+' if diff > 0 else '−'}{mag} BPM"

    c0, c1 = cur.get("camelot"), e.get("camelot")
    key_chip = (c1 if c1 == c0 or not c0 else f"{c0}→{c1}") if c1 else None
    g = e.get("genres") or []
    genre_chip = g[0][0].split("---")[-1] if g else None

    if d.axis == "bpm":
        order = [bpm_chip, key_chip, genre_chip]
    elif d.axis == "similar":
        order = [genre_chip, bpm_chip, key_chip]
    else:
        order = [axis_chip, bpm_chip, key_chip, genre_chip]
    return [c for c in order if c]


def public_track(e: dict) -> dict:
    """The fields the UI needs; never the embedding."""
    g = e.get("genres") or []
    heads = e.get("heads") or {}
    return {
        "id": e["id"],
        "title": e.get("title"),
        "artist": e.get("artist"),
        "album": e.get("album"),
        "bpm": e.get("bpm"),
        "bpm_estimated": bool(e.get("bpm_estimated")),
        "key": e.get("camelot"),
        "key_estimated": bool(e.get("key_estimated")),
        "energy": e.get("energy"),
        "rating": e.get("rating") or 0,
        "genres": [x[0].split("---")[-1] for x in g[:3]],
        "vocal": heads.get("voice"),
        "crate": e.get("crate"),
        "source": e.get("source"),
        "duration": e.get("duration"),
        "path": e.get("path"),
    }
