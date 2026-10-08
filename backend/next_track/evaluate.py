"""How good are the suggestions? Two independent checks.

**Mixing rules** (the main judge). From a few hundred starting tracks, look at
what the compass actually puts on screen and score it against rules that hold
whoever is mixing: harmonic key moves, small tempo changes, every direction
really moving that way, cards not left empty.

**Taste check** (a sanity check, not a target). Your recent rekordbox sessions
say which record you actually played after which. Centred on the first track,
does the compass put the one you chose anywhere near the top? It measures
whether suggestions stay in your taste. Your older mixes aren't used, and a
transition that breaks the mixing rules isn't something to copy, so this is
reported alongside the rules and never optimised on its own.
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from . import engine
from .index import LibraryIndex, track_id

TASTE_SINCE = datetime(2026, 2, 1)
MIN_SESSION_TRACKS = 4
GAP_SECONDS = (90, 15 * 60)   # shorter = skipped/previewed, longer = a break


def recent_transitions(index: LibraryIndex, since: datetime = TASTE_SINCE) -> list[tuple]:
    """(from_id, to_id, earlier_ids) for consecutive plays in recent sessions
    where both tracks are in the analysed library."""
    from pyrekordbox.db6 import tables as t

    with index.rekordbox._lock:
        db = index.rekordbox._fresh()
        rows = db.query(t.DjmdSongHistory).all()
        sessions = defaultdict(list)
        for r in rows:
            c = r.Content
            if c is None or not c.FolderPath or r.created_at is None:
                continue
            sessions[r.HistoryID].append((r.created_at, c.FolderPath))

    analysed = set(index.lib_ids)
    out = []
    for plays in sessions.values():
        if len(plays) < MIN_SESSION_TRACKS:
            continue
        plays.sort()
        if plays[0][0] < since:
            continue
        ids = [track_id(p) for _, p in plays]
        for i in range(1, len(plays)):
            gap = (plays[i][0] - plays[i - 1][0]).total_seconds()
            a, b = ids[i - 1], ids[i]
            if a == b or not (GAP_SECONDS[0] <= gap <= GAP_SECONDS[1]):
                continue
            if a in analysed and b in analysed:
                out.append((a, b, set(ids[:i - 1])))
    return out


def follows_rules(index: LibraryIndex, a: str, b: str) -> bool:
    """Would a careful DJ call this a clean transition? Tempo within 6% and a
    harmonically compatible key."""
    from . import keys
    A, B = index.entries[a], index.entries[b]
    if A.get("bpm") and B.get("bpm") and abs(B["bpm"] - A["bpm"]) / A["bpm"] > 0.06:
        return False
    return keys.compatibility(A.get("camelot"), B.get("camelot")) >= 0.5


def taste_check(index: LibraryIndex, transitions: list[tuple],
                params: engine.Params = engine.DEFAULT_PARAMS) -> dict:
    """Taste check, on the transitions that follow the mixing rules only
    (copying a key clash isn't the goal). See _taste for the measures."""
    clean = [t for t in transitions if follows_rules(index, t[0], t[1])]
    out = _taste(index, clean, params)
    out["all_transitions"] = len(transitions)
    out.update(sound_fit(index, transitions, params))
    return out


def sound_fit(index: LibraryIndex, transitions: list[tuple],
              params: engine.Params = engine.DEFAULT_PARAMS) -> dict:
    """Does the next track *sound* like it follows? For every transition that
    stays within 6% tempo (key clashes included: this is about sound, not key
    choice), the percentile of the real next track among all tempo-compatible
    candidates, ranked by the fit signal alone. 1.0 = ranked first."""
    pcts = []
    for a, b, earlier in transitions:
        A, B = index.entries[a], index.entries[b]
        if A.get("bpm") and B.get("bpm") and abs(B["bpm"] - A["bpm"]) / A["bpm"] > 0.06:
            continue
        ctx = engine.prepare(index, a, earlier, params)
        if b not in ctx.ids:
            continue
        rb = ctx.ids.index(b)
        cand = ctx.pool & (np.abs(np.nan_to_num(ctx.rel, nan=1.0)) <= 0.06)
        cand[rb] = True
        below = np.sum(cand & (ctx.fit_raw < ctx.fit_raw[rb]))
        pcts.append(below / max(cand.sum() - 1, 1))
    return {"fit_transitions": len(pcts),
            "fit_pct": round(float(np.mean(pcts)), 3) if pcts else None,
            "fit_top10pct": round(float(np.mean(np.array(pcts) >= 0.9)), 3) if pcts else None}


def _taste(index: LibraryIndex, transitions: list[tuple],
           params: engine.Params = engine.DEFAULT_PARAMS) -> dict:
    """Where does your actual next track land?

    - on_screen: it's one of the 24 tracks the compass shows,
    - top3_any: it's in the top 3 of at least one direction (before the
      one-track-per-direction de-duplication),
    - best_rank: its best rank across directions (median reported).
    """
    on_screen = top3 = 0
    best_ranks = []
    dirs = [engine.DIRECTIONS[d] for d in engine.DEFAULT_DIRECTIONS]
    for a, b, earlier in transitions:
        res = engine.suggest(index, a, exclude_ids=earlier, params=params)
        shown = {t["id"] for d in res["directions"] for t in d["tracks"]}
        on_screen += b in shown
        ctx = engine.prepare(index, a, earlier, params)
        if b not in ctx.ids:
            continue
        rb = ctx.ids.index(b)
        best = None
        for d in dirs:
            if d.axis == "bpm" and not ctx.bpm0:
                continue
            mask, score = engine.direction_scores(ctx, d)
            if not mask[rb]:
                continue
            rank = int(np.sum(mask & (score > score[rb])))
            best = rank if best is None else min(best, rank)
        if best is not None:
            best_ranks.append(best)
            top3 += best < 3
        else:
            best_ranks.append(len(ctx.ids))
    n = max(len(transitions), 1)
    return {
        "transitions": len(transitions),
        "on_screen": round(on_screen / n, 3),
        "top3_any": round(top3 / n, 3),
        "median_best_rank": float(np.median(best_ranks)) if best_ranks else None,
        "reachable": round(np.mean([r < len(index.lib_ids) for r in best_ranks]), 3) if best_ranks else None,
    }


def rule_check(index: LibraryIndex, params: engine.Params = engine.DEFAULT_PARAMS,
               n_starts: int = 250, seed: int = 7) -> dict:
    """Score what's on screen against mixing rules, from random starting tracks."""
    rng = np.random.default_rng(seed)
    starts = rng.choice(index.lib_ids, min(n_starts, len(index.lib_ids)), replace=False)
    key, bpm_jump, harmonic, wrong_way, empty, picks, fold = [], [], 0, 0, 0, 0, 0
    cards = 0
    for tid in starts:
        res = engine.suggest(index, tid, params=params)
        cur = res["current"]
        for d in res["directions"]:
            cards += 1
            if not d["tracks"]:
                empty += 1
                continue
            t = d["tracks"][0]
            picks += 1
            key.append(t["key_compat"])
            harmonic += t["key_compat"] >= 0.85
            if any(c.startswith(("½×", "2×")) for c in t["reasons"]):
                fold += 1
            elif cur["bpm"] and t["bpm"]:
                bpm_jump.append(abs(t["bpm"] - cur["bpm"]) / cur["bpm"])
            if t["shift"] is not None and t["shift"] <= 0:
                wrong_way += 1
    return {
        "starts": len(starts),
        "harmonic": round(harmonic / max(picks, 1), 3),
        "mean_key": round(float(np.mean(key)), 3) if key else None,
        "median_bpm_jump_pct": round(float(np.median(bpm_jump)) * 100, 2) if bpm_jump else None,
        "empty_cards": round(empty / max(cards, 1), 3),
        "wrong_way": round(wrong_way / max(picks, 1), 3),
        "folded_tempo": round(fold / max(picks, 1), 3),
    }


def evaluate(index: LibraryIndex, params: engine.Params = engine.DEFAULT_PARAMS,
             transitions: Optional[list] = None) -> dict:
    if transitions is None:
        transitions = recent_transitions(index)
    return {"rules": rule_check(index, params), "taste": taste_check(index, transitions, params)}
