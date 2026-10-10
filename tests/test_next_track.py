"""Tests for Next Track's key handling, suggestion rules and evaluation.

Run with the standard library, no extra packages:
    .venv1/bin/python -m unittest discover tests
"""
import threading
import unittest

import numpy as np

from backend.next_track import engine, keys
from backend.next_track.evaluate import follows_rules


class KeyTests(unittest.TestCase):
    def test_musical_keys_to_camelot(self):
        self.assertEqual(keys.to_camelot("Am"), "8A")
        self.assertEqual(keys.to_camelot("C"), "8B")
        self.assertEqual(keys.to_camelot("F#m"), "11A")
        self.assertEqual(keys.to_camelot("Ebm"), "2A")
        self.assertEqual(keys.to_camelot("Db"), "3B")

    def test_camelot_and_open_key_pass_through(self):
        self.assertEqual(keys.to_camelot("8A"), "8A")
        self.assertEqual(keys.to_camelot("12b"), "12B")
        # Open Key 1m is A minor (8A); 3m is B minor (10A); 1d is C major (8B).
        self.assertEqual(keys.to_camelot("1m"), "8A")
        self.assertEqual(keys.to_camelot("3m"), "10A")
        self.assertEqual(keys.to_camelot("1d"), "8B")

    def test_unreadable_keys(self):
        self.assertIsNone(keys.to_camelot(None))
        self.assertIsNone(keys.to_camelot(""))
        self.assertIsNone(keys.to_camelot("H#"))

    def test_compatibility(self):
        self.assertEqual(keys.compatibility("8A", "8A"), 1.0)
        self.assertEqual(keys.compatibility("8A", "9A"), 0.9)
        self.assertEqual(keys.compatibility("12A", "1A"), 0.9)   # wraps round
        self.assertEqual(keys.compatibility("8A", "8B"), 0.85)
        self.assertEqual(keys.compatibility("8A", "3B"), 0.15)
        self.assertEqual(keys.compatibility(None, "8A"), 0.5)

    def test_is_minor(self):
        self.assertTrue(keys.is_minor("5A"))
        self.assertFalse(keys.is_minor("5B"))
        self.assertIsNone(keys.is_minor(None))


class FakeIndex:
    """Just enough of LibraryIndex for the engine: a synthetic library where
    track i's tempo, key and axis positions are known."""

    def __init__(self, n=200, seed=0, extra=None):
        rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self.lib_ids = [f"t{i:03d}" for i in range(n)]
        cams = [f"{k}{l}" for k in range(1, 13) for l in "AB"]
        self.entries = {}
        for i, tid in enumerate(self.lib_ids):
            self.entries[tid] = {
                "id": tid, "title": f"Track {i}", "artist": f"Artist {i % 40}",
                "bpm": float(rng.choice([124, 125, 126, 127, 128, 130, 132, 140, 87, 174])),
                "camelot": str(rng.choice(cams)), "genres": [["Electronic---House", 0.5]],
            }
        X = rng.normal(size=(n, 16))
        self.Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.axis_pct = {a: rng.random(n) for a in ("energy", "dark", "vocal", "deep", "electronic")}
        self._extra = extra or {}

    def features_for(self, tid):
        r = self.lib_ids.index(tid)
        return {"vec": self.Xn[r], "pct": {a: float(v[r]) for a, v in self.axis_pct.items()},
                "axes": {}, "row": r}

    def get(self, tid):
        return self.entries.get(tid)

    def extra_similarities(self, tid):
        return self._extra


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.idx = FakeIndex()
        self.start = next(t for t in self.idx.lib_ids if self.idx.entries[t]["bpm"] == 126)

    def shown(self, res):
        return [t for d in res["directions"] for t in d["tracks"]]

    def test_every_default_direction_is_returned(self):
        res = engine.suggest(self.idx, self.start)
        self.assertEqual([d["id"] for d in res["directions"]], engine.DEFAULT_DIRECTIONS)

    def test_no_track_shown_twice(self):
        ids = [t["id"] for t in self.shown(engine.suggest(self.idx, self.start))]
        self.assertEqual(len(ids), len(set(ids)))

    def test_current_and_played_tracks_are_never_suggested(self):
        played = set(self.idx.lib_ids[:30])
        res = engine.suggest(self.idx, self.start, exclude_ids=played)
        ids = {t["id"] for t in self.shown(res)}
        self.assertNotIn(self.start, ids)
        self.assertFalse(ids & played)

    def test_tempo_stays_in_reach(self):
        res = engine.suggest(self.idx, self.start)
        bpm0 = self.idx.entries[self.start]["bpm"]
        for d in res["directions"]:
            for t in d["tracks"]:
                limit = 0.09 if d["id"] in ("faster", "slower") else engine.DEFAULT_PARAMS.relaxed_bpm_window
                self.assertLessEqual(abs(t["bpm"] - bpm0) / bpm0, limit + 1e-9, d["id"])

    def test_faster_and_slower_move_the_right_way(self):
        res = {d["id"]: d for d in engine.suggest(self.idx, self.start)["directions"]}
        bpm0 = self.idx.entries[self.start]["bpm"]
        for t in res["faster"]["tracks"]:
            self.assertGreater(t["bpm"], bpm0)
        for t in res["slower"]["tracks"]:
            self.assertLess(t["bpm"], bpm0)

    def test_axis_directions_move_along_their_axis(self):
        res = engine.suggest(self.idx, self.start)
        for d in res["directions"]:
            if d["id"] in ("energy_up", "darker", "vocal", "deeper", "energy_down"):
                for t in d["tracks"]:
                    self.assertGreater(t["shift"], 0, d["id"])

    def test_extreme_track_explains_empty_direction(self):
        idx = FakeIndex(n=40, seed=3)
        tid = idx.lib_ids[0]
        idx.axis_pct["dark"][:] = 0.0
        idx.axis_pct["dark"][0] = 1.0     # the darkest track there is
        res = {d["id"]: d for d in engine.suggest(idx, tid)["directions"]}
        self.assertEqual(res["darker"]["tracks"], [])
        self.assertEqual(res["darker"]["empty_reason"], "Already one of your darkest")

    def test_halftime_folds_only_when_enabled(self):
        start = next(t for t in self.idx.lib_ids if self.idx.entries[t]["bpm"] == 174)
        off = engine.prepare(self.idx, start, params=engine.DEFAULT_PARAMS.with_(halftime=False))
        on = engine.prepare(self.idx, start, params=engine.DEFAULT_PARAMS.with_(halftime=True))
        r87 = next(i for i, t in enumerate(off.ids) if off.ents[i]["bpm"] == 87)
        self.assertEqual(off.folded[r87], 1.0)
        self.assertEqual(on.folded[r87], 2.0)
        self.assertAlmostEqual(on.eff_bpm[r87], 174.0)

    def test_fit_blends_extra_signals(self):
        n = len(self.idx.lib_ids)
        favour = np.zeros(n)
        favour[5] = 10.0                   # one candidate fits far better
        idx = FakeIndex(extra={"transition": favour})
        start = self.start
        plain = engine.prepare(idx, start, params=engine.DEFAULT_PARAMS)
        blended = engine.prepare(idx, start, params=engine.DEFAULT_PARAMS.with_(w_transition=1.0))
        if plain.pool[5]:
            self.assertGreaterEqual(blended.fit_rank[5], plain.fit_rank[5])


class RuleFilterTests(unittest.TestCase):
    def test_follows_rules(self):
        idx = FakeIndex(n=3)
        a, b, c = idx.lib_ids
        idx.entries[a].update(bpm=126.0, camelot="8A")
        idx.entries[b].update(bpm=127.0, camelot="9A")    # clean
        idx.entries[c].update(bpm=127.0, camelot="3B")    # key clash
        self.assertTrue(follows_rules(idx, a, b))
        self.assertFalse(follows_rules(idx, a, c))
        idx.entries[b]["bpm"] = 140.0                     # tempo jump
        self.assertFalse(follows_rules(idx, a, b))


class DeepAxisTests(unittest.TestCase):
    """The "deep" axis is learnt from Discogs styles over the fingerprint."""

    def make(self, n_deep, n_other, n_unlabelled=20, seed=0):
        from backend.next_track.index import LibraryIndex
        rng = np.random.default_rng(seed)
        idx = LibraryIndex.__new__(LibraryIndex)
        n = n_deep + n_other + n_unlabelled
        ids = [f"t{i}" for i in range(n)]
        Z = rng.normal(size=(n, 32))
        Z[:n_deep, 0] += 3.0                     # deep tracks share a direction
        styles = {t: {"Deep House"} for t in ids[:n_deep]}
        styles.update({t: {"Techno"} for t in ids[n_deep:n_deep + n_other]})
        idx._discogs_styles = lambda: styles
        idx._mu, idx._sd = np.zeros(64), np.ones(64)
        return idx, ids, Z

    def test_deep_tracks_score_higher(self):
        idx, ids, Z = self.make(40, 60)
        v = idx._derive_deep(ids, Z)
        self.assertGreater(v[:40].mean(), v[40:100].mean() + 1.0)
        # a track outside the library is scored by the same model
        self.assertEqual(idx._deep_score(np.concatenate([Z[0], Z[0]])).shape, (1,))

    def test_too_few_examples_falls_back(self):
        idx, ids, Z = self.make(10, 60)
        self.assertIsNone(idx._derive_deep(ids, Z))
        self.assertIsNone(idx._deep_score(np.zeros(64)))


if __name__ == "__main__":
    unittest.main()
