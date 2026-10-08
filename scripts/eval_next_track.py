"""
eval_next_track.py — score Next Track's suggestions, and compare settings.

Two checks (see backend/next_track/evaluate.py):
  rules  what the compass shows, judged by mixing rules (the main judge)
  taste  where your actual next track landed, from rekordbox sessions since
         Feb 2026 (a sanity check that it stays in your taste)

    .venv1/bin/python scripts/eval_next_track.py            # current settings
    .venv1/bin/python scripts/eval_next_track.py --variants # compare variants
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.next_track import engine  # noqa: E402
from backend.next_track.evaluate import evaluate, recent_transitions  # noqa: E402
from backend.next_track.index import LibraryIndex  # noqa: E402

P = engine.DEFAULT_PARAMS
BASE = engine.Params()   # the original settings, for comparison
VARIANTS = {
    "original": BASE,
    "current": P,
    "halftime": BASE.with_(halftime=True),
    "crate 0.25": BASE.with_(w_crate=0.25),
    "crate 0.5": BASE.with_(w_crate=0.5),
    "crate 1.0": BASE.with_(w_crate=1.0),
    "transition 0.5": BASE.with_(w_transition=0.5),
    "transition 1.0": BASE.with_(w_transition=1.0),
    "transition 2.0": BASE.with_(w_transition=2.0),
    "trans 1 + crate .5": BASE.with_(w_transition=1.0, w_crate=0.5),
}


def row(name, r):
    ru, ta = r["rules"], r["taste"]
    return (f"{name:18s} harmonic {ru['harmonic']:.2f}  key {ru['mean_key']:.2f}  "
            f"bpm±{ru['median_bpm_jump_pct']:.1f}%  empty {ru['empty_cards']:.2f}  "
            f"folded {ru['folded_tempo']:.2f}  |  on-screen {ta['on_screen']:.2f}  "
            f"top3 {ta['top3_any']:.2f}  rank {ta['median_best_rank']}  |  "
            f"sound-fit {ta['fit_pct']:.3f}  top10% {ta['fit_top10pct']:.2f}")


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--only", help="only variants whose name contains this")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    idx = LibraryIndex()
    trans = recent_transitions(idx)
    from backend.next_track.evaluate import follows_rules
    clean = sum(follows_rules(idx, a, b) for a, b, _ in trans)
    print(f"{len(idx.lib_ids)} analysed tracks, {len(trans)} recent transitions, "
          f"{clean} of them follow the mixing rules (the taste check uses those)\n")
    names = VARIANTS if args.variants else {"current": P}
    if args.only:
        names = {k: v for k, v in names.items() if args.only in k or k == "original"}
    out = {}
    for name, params in names.items():
        try:
            out[name] = evaluate(idx, params, trans)
        except Exception as exc:  # a variant whose data isn't built yet
            print(f"{name:18s} skipped: {exc}")
            continue
        print(row(name, out[name]), flush=True)
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
