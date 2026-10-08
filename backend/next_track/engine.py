"""Directional suggestions: from the track on air, where can you go next?

Every direction answers the same question with a different push:

    "Which mixable track fits best after this one, *but* noticeably more X?"

"Mixable" is a hard filter first: tempo within reach of the pitch fader (or
half/double time), not the track that's playing, not anything already played
this session. Inside that pool each direction scores candidates on:

- fit: how well the candidate follows this track. A blend of whole-track
  similarity, how this track's outro sounds against the candidate's intro,
  and whether they'd sit in the same crate. Ranked within the pool so the
  scale is comparable from track to track.
- shift: how far the candidate moves along the direction's axis, measured in
  library percentiles (so "darker" means darker than this track relative to
  *your* music), saturating so the pick is a step, not a leap,
- key: Camelot-wheel compatibility,
- tempo: closeness in BPM.

The weights live in :class:`Params` so ``scripts/eval_next_track.py`` can
compare settings against your own recent sets and the mixing rules.

Directions are filled in priority order and a track is used at most once, so
the screen never shows the same record twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class Params:
    # How "fit" is built from the similarity signals (renormalised over the
    # signals the index actually has).
    w_whole: float = 1.0          # whole-track embedding similarity
    w_transition: float = 0.0     # this track's outro vs the candidate's intro
    w_crate: float = 0.0          # would they sit in the same crate?
    # Axis directions: fit / shift / key / tempo.
    axis_weights: tuple = (0.45, 0.30, 0.15, 0.10)
    # Faster/slower: fit / tempo step / key.
    bpm_weights: tuple = (0.50, 0.25, 0.25)
    # Closest blend: fit / key / tempo.
    closest_weights: tuple = (0.65, 0.25, 0.10)
    min_shift: float = 0.10       # percentile points along the axis
    full_shift: float = 0.35      # shift beyond this earns no extra credit
    bpm_window: float = 0.06      # +/- for non-tempo directions
    relaxed_min_shift: float = 0.03
    relaxed_bpm_window: float = 0.09
    bpm_step: tuple = (1.5, 10.0)  # faster/slower must move this many BPM
    bpm_step_pct: float = 0.09     # ...and no more than this
    halftime: bool = False         # allow 87 under 174 and vice versa
    halftime_penalty: float = 0.08
    min_key: float = 0.5           # prefer picks at least this compatible

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


DEFAULT_PARAMS = Params()


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


@dataclass
class Context:
    """Everything the directions share for one starting track."""
    cur: dict
    feats: dict
    ids: list
    ents: list
    pct: dict
    pool: np.ndarray
    fit_raw: np.ndarray          # combined similarity, before ranking
    fit_rank: np.ndarray         # ranked within the pool, 0..1
    whole_sim: np.ndarray
    key_score: np.ndarray
    bpm0: Optional[float]
    eff_bpm: np.ndarray          # candidate BPM after half/double folding
    rel: np.ndarray              # (eff_bpm - bpm0) / bpm0
    folded: np.ndarray           # 1, 2 or 0.5: how the tempo was folded
    bpm_close: np.ndarray
    params: Params = field(default_factory=Params)


def prepare(index: LibraryIndex, current_id: str,
            exclude_ids: Optional[set[str]] = None,
            params: Params = DEFAULT_PARAMS) -> Context:
    feats = index.features_for(current_id)
    cur = index.get(current_id)
    if feats is None or cur is None:
        raise KeyError(current_id)

    with index._lock:
        ids = list(index.lib_ids)
        ents = [index.entries[i] for i in ids]
        Xn = index.Xn
        pct = {a: index.axis_pct[a] for a in index.axis_pct}
        extra = index.extra_similarities(current_id)

    exclude = set(exclude_ids or ())
    exclude.add(current_id)
    exclude_names = {_norm_title(cur)}
    for x in exclude:
        e = index.get(x)
        if e:
            exclude_names.add(_norm_title(e))
    pool = np.array([(i not in exclude) and (_norm_title(e) not in exclude_names)
                     for i, e in zip(ids, ents)], dtype=bool)

    whole = Xn @ feats["vec"] if len(ids) else np.zeros(0)
    parts = [(params.w_whole, whole)]
    if params.w_transition and extra.get("transition") is not None:
        parts.append((params.w_transition, extra["transition"]))
    if params.w_crate and extra.get("crate") is not None:
        parts.append((params.w_crate, extra["crate"]))
    total = sum(w for w, _ in parts) or 1.0
    # Each signal is ranked first so one with a wider spread can't dominate.
    fit_raw = np.zeros(len(ids))
    for w, s in parts:
        r = np.zeros(len(ids))
        if pool.any():
            r[pool] = _rank01(s[pool])
        fit_raw += (w / total) * r
    fit_rank = np.zeros(len(ids))
    if pool.any():
        fit_rank[pool] = _rank01(fit_raw[pool])

    bpm0 = cur.get("bpm")
    cam0 = cur.get("camelot")
    bpm = np.array([e.get("bpm") or np.nan for e in ents], dtype=np.float64)
    key_score = np.array([keys.compatibility(e.get("camelot"), cam0) for e in ents])

    folded = np.ones(len(ids))
    eff = bpm.copy()
    if bpm0:
        rel = (bpm - bpm0) / bpm0
        if params.halftime:
            for f in (2.0, 0.5):
                alt = (bpm * f - bpm0) / bpm0
                better = np.abs(np.nan_to_num(alt, nan=9)) < np.abs(np.nan_to_num(rel, nan=9))
                rel = np.where(better, alt, rel)
                folded = np.where(better, f, folded)
            eff = bpm * folded
        d = np.abs(np.nan_to_num(eff, nan=bpm0 * 1.5) - bpm0)
        bpm_close = np.clip(1 - d / 8.0, 0, 1)
    else:
        rel = np.zeros(len(ids))
        bpm_close = np.full(len(ids), 0.5)

    return Context(cur=cur, feats=feats, ids=ids, ents=ents, pct=pct, pool=pool,
                   fit_raw=fit_raw, fit_rank=fit_rank, whole_sim=whole,
                   key_score=key_score, bpm0=bpm0, eff_bpm=eff, rel=rel,
                   folded=folded, bpm_close=bpm_close, params=params)


def direction_scores(ctx: Context, d: Direction, alternates: int = 2):
    """(eligible mask, score) for one direction. Eligible already applies the
    tempo, shift and preferred-key rules."""
    p = ctx.params
    n = len(ctx.ids)
    base = ctx.pool
    absrel = np.abs(np.nan_to_num(ctx.rel, nan=1.0))
    steady = absrel <= p.bpm_window if ctx.bpm0 else np.ones(n, dtype=bool)
    penalty = np.where(ctx.folded != 1.0, p.halftime_penalty, 0.0)

    if d.axis == "bpm":
        if not ctx.bpm0:
            return np.zeros(n, dtype=bool), np.zeros(n)
        step = d.sign * (ctx.eff_bpm - ctx.bpm0)
        ok = (step >= p.bpm_step[0]) & (step <= p.bpm_step[1]) & (absrel <= p.bpm_step_pct)
        mask = base & ok
        shift = np.clip(1 - np.abs(np.nan_to_num(step) - 4.0) / 6.0, 0, 1)
        wf, ws, wk = p.bpm_weights
        score = wf * ctx.fit_rank + ws * shift + wk * ctx.key_score
    elif d.axis == "similar":
        mask = base & steady
        if mask.sum() < 1 + alternates and ctx.bpm0:
            mask = base & (absrel <= p.relaxed_bpm_window)
        wf, wk, wt = p.closest_weights
        score = wf * ctx.fit_rank + wk * ctx.key_score + wt * ctx.bpm_close
    else:
        delta = d.sign * (ctx.pct[d.axis] - ctx.feats["pct"][d.axis])
        mask = base & steady & (delta >= p.min_shift)
        if mask.sum() < 1 + alternates:
            # The track already sits near the end of this axis (the darkest
            # record in the crate can't get much darker). Accept a smaller
            # push and a little more tempo room rather than show nothing.
            wide = absrel <= p.relaxed_bpm_window if ctx.bpm0 else steady
            mask = base & wide & (delta >= p.relaxed_min_shift)
        shift = np.clip(delta / p.full_shift, 0, 1)
        wf, ws, wk, wt = p.axis_weights
        score = wf * ctx.fit_rank + ws * shift + wk * ctx.key_score + wt * ctx.bpm_close

    score = score - penalty
    # Prefer harmonically safe picks; fall back to anything in range.
    strict = mask & (ctx.key_score >= p.min_key)
    return (strict if strict.sum() >= 1 + alternates else mask), score


def suggest(index: LibraryIndex, current_id: str,
            exclude_ids: Optional[set[str]] = None,
            directions: Optional[list[str]] = None,
            alternates: int = 2,
            params: Params = DEFAULT_PARAMS) -> dict:
    """Suggestions for every requested direction from ``current_id``."""
    directions = [d for d in (directions or DEFAULT_DIRECTIONS) if d in DIRECTIONS]
    ctx = prepare(index, current_id, exclude_ids, params)

    used: set[str] = set()
    used_names: set[str] = set()
    results: dict[str, dict] = {}

    for did in sorted(directions, key=_PRIORITY.index):
        d = DIRECTIONS[did]
        if d.axis == "bpm" and not ctx.bpm0:
            results[did] = {"id": did, "label": d.label, "tracks": [],
                            "empty_reason": "No BPM for the current track"}
            continue
        mask, score = direction_scores(ctx, d, alternates)
        order = np.argsort(-np.where(mask, score, -np.inf))
        picks = []
        for r in order:
            if not mask[r]:
                break
            name = _norm_title(ctx.ents[r])
            if ctx.ids[r] in used or name in used_names:
                continue
            picks.append(r)
            used_names.add(name)
            if len(picks) >= 1 + alternates:
                break
        if picks:
            used.add(ctx.ids[picks[0]])

        tracks = []
        for r in picks:
            e = ctx.ents[r]
            tracks.append({
                **public_track(e),
                "match": round(float(ctx.whole_sim[r]), 3),
                "score": round(float(score[r]), 3),
                "key_compat": round(float(ctx.key_score[r]), 2),
                "shift": (round(float(d.sign * (ctx.pct[d.axis][r] - ctx.feats["pct"][d.axis])), 3)
                          if d.axis in ctx.pct else None),
                "reasons": _reasons(d, ctx, r),
            })
        results[did] = {"id": did, "label": d.label, "tracks": tracks,
                        "empty_reason": None if tracks else _empty_reason(d, ctx.feats)}

    return {
        "current": {**public_track(ctx.cur), "pct": ctx.feats["pct"]},
        "directions": [results[d] for d in directions if d in results],
        "pool_size": int(ctx.pool.sum()),
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


def _reasons(d: Direction, ctx: Context, r: int) -> list[str]:
    """Short chips explaining a pick, most relevant to the direction first."""
    e, cur, feats, pct = ctx.ents[r], ctx.cur, ctx.feats, ctx.pct
    words = {"energy": ("energy", "calmer"), "dark": ("darker", "lighter"),
             "vocal": ("vocal", "less vocal"), "electronic": ("electronic", "organic")}
    axis_chip = None
    if d.axis in words:
        delta = pct[d.axis][r] - feats["pct"][d.axis]
        axis_chip = f"{words[d.axis][0 if delta > 0 else 1]} +{round(abs(delta) * 100)}"

    bpm_chip = None
    b0, b1 = cur.get("bpm"), e.get("bpm")
    if b1:
        fold = ctx.folded[r]
        if fold != 1.0:
            bpm_chip = f"{'½×' if fold == 0.5 else '2×'} {b1:.0f} BPM"
        else:
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
