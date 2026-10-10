"""
bakeoff_embeddings.py — which audio fingerprint best describes your music?

Uses the Discogs styles fetched by fetch_discogs_tags.py as an answer key.
For each candidate fingerprint, a simple classifier learns to predict styles
from the fingerprint on 4/5 of your matched tracks and is scored on the 1/5
it never saw (5-fold, every track held out once). Higher is better:

  auc   mean ROC-AUC over styles (0.5 = guessing, 1.0 = perfect)
  hit@1 the top predicted style is one of the track's real Discogs styles
  hit@3 ...one of the top three is

No labelling needed: Discogs is the judge. Styles with fewer than MIN_TRACKS
examples are skipped, and only confident Discogs matches are used.

Text-and-audio fingerprints from embed_text_audio.py (emb_<model>.npz) are
included when present, plus a zero-shot row: each style is scored by how
close the track sits to the text "<style> music", with no training at all.
Only tracks every fingerprint covers are used, so rows are comparable.

    .venv1/bin/python scripts/bakeoff_embeddings.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.next_track import paths  # noqa: E402
from backend.next_track.index import LibraryIndex  # noqa: E402

MIN_TRACKS = 15
MIN_CONFIDENCE = 0.75
TEXT_AUDIO = ("muq", "clap")
# The "Deeper" dial's answer key: tracks Discogs files under any of these.
DEEP_STYLES = {"Deep House", "Deep Techno", "Dub Techno"}
SEEDS = (0, 1, 2)


def load_text_audio():
    """{model: (ids -> vector, words, word vectors)} for each emb_<model>.npz."""
    out = {}
    for name in TEXT_AUDIO:
        f = paths.data_dir() / f"emb_{name}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            out[name] = ({str(i): v for i, v in zip(z["ids"], z["emb"])},
                         [str(w) for w in z["words"]], z["word_emb"])
    return out


def load_labels(idx, extra):
    raw = json.loads((paths.data_dir() / "discogs.json").read_text())
    rows = [(tid, v["styles"]) for tid, v in raw.items()
            if v.get("status") == "ok" and v.get("confidence", 0) >= MIN_CONFIDENCE
            and v.get("styles") and tid in idx._row_of
            and all(tid in vecs for vecs, _, _ in extra.values())]
    counts = Counter(s for _, st in rows for s in st)
    styles = sorted(s for s, n in counts.items() if n >= MIN_TRACKS)
    rows = [(tid, [s for s in st if s in styles]) for tid, st in rows]
    rows = [(tid, st) for tid, st in rows if st]
    Y = np.array([[s in st for s in styles] for _, st in rows], dtype=int)
    return [tid for tid, _ in rows], styles, Y


def feature_sets(idx, ids, extra):
    rows = [idx._row_of[t] for t in ids]
    ents = [idx.entries[t] for t in ids]
    X = np.stack([idx._emb[t] for t in ids]).astype(np.float32)
    half = X.shape[1] // 2
    sets = {
        "EffNet mean+std (current)": X,
        "EffNet mean only": X[:, :half],
    }
    # A few tracks lack intro/outro or mood/theme data: fill with the mean.
    if sum(t in idx._intro and t in idx._outro for t in ids) > 0.9 * len(ids):
        def edge(d):
            dim = len(next(iter(d.values())))
            A = np.stack([d.get(t, np.full(dim, np.nan)) for t in ids])
            return np.where(np.isnan(A), np.nanmean(A, axis=0), A)
        sets["EffNet mid+intro+outro"] = np.hstack([X[:, :half], edge(idx._intro),
                                                    edge(idx._outro)])
    heads = sorted(ents[0]["heads"])
    sets["7 mood/genre heads only"] = np.array([[e["heads"][h] for h in heads] for e in ents])
    if sum(bool(e.get("moods")) for e in ents) > 0.9 * len(ents):
        mk = sorted(next(e["moods"] for e in ents if e.get("moods")))
        M = np.array([[(e.get("moods") or {}).get(k, np.nan) for k in mk] for e in ents])
        M = np.where(np.isnan(M), np.nanmean(M, axis=0), M)
        sets["Jamendo mood/theme (56)"] = M
        sets["EffNet mean + mood/theme"] = np.hstack([X[:, :half], M * 10])
    for name, (vecs, _, _) in extra.items():
        T = np.stack([vecs[t] for t in ids])
        sets[f"{name} (text+audio)"] = T
        sets[f"EffNet mean + {name}"] = np.hstack([X[:, :half], T])
    return sets


def zero_shot(T, words, W, styles, Y):
    """Score styles straight from text similarity, no training. Each style's
    scores are standardised over the library so styles are comparable."""
    lookup = {w.lower(): i for i, w in enumerate(words)}
    cols = [j for j, s in enumerate(styles) if s.lower() in lookup]
    if not cols:
        return None
    from sklearn.metrics import roc_auc_score
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    S = T @ Wn[[lookup[styles[j].lower()] for j in cols]].T
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-9)
    Yc = Y[:, cols]
    aucs = {styles[j]: roc_auc_score(Yc[:, k], S[:, k]) for k, j in enumerate(cols)
            if 0 < Yc[:, k].sum() < len(Yc)}
    order = np.argsort(-S, axis=1)
    hit1 = np.mean([Yc[i, order[i, 0]] for i in range(len(Yc))])
    hit3 = np.mean([Yc[i, order[i, :3]].any() for i in range(len(Yc))])
    return float(np.mean(list(aucs.values()))), float(hit1), float(hit3), len(cols), aucs


def evaluate(X, Y, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    P = np.zeros(Y.shape, dtype=float)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if Xtr.shape[1] > 128:   # keep it fair: same capacity for every fingerprint
            pca = PCA(min(128, len(Xtr) - 1), random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        for j in range(Y.shape[1]):
            if Y[tr, j].min() == Y[tr, j].max():
                continue
            m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
            m.fit(Xtr, Y[tr, j])
            P[te, j] = m.predict_proba(Xte)[:, 1]
    aucs = np.array([roc_auc_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y)
                     else np.nan for j in range(Y.shape[1])])
    order = np.argsort(-P, axis=1)
    hit1 = np.mean([Y[i, order[i, 0]] for i in range(len(Y))])
    hit3 = np.mean([Y[i, order[i, :3]].any() for i in range(len(Y))])
    return float(np.nanmean(aucs)), float(hit1), float(hit3), aucs


def evaluate_seeds(X, Y):
    runs = [evaluate(X, Y, seed=s) for s in SEEDS]
    return (*(float(np.mean([r[k] for r in runs])) for k in range(3)),
            np.nanmean([r[3] for r in runs], axis=0))


def dial_check(idx, ids, extra, sets):
    """How well does each candidate signal rank "deep" records above the rest?
    Fixed axes (the index's deep/dark/electronic values) and zero-shot text
    scores are used as they are; fingerprints get a classifier, out-of-fold."""
    from sklearn.metrics import roc_auc_score
    raw = json.loads((paths.data_dir() / "discogs.json").read_text())
    y = np.array([bool(DEEP_STYLES & set(raw[t]["styles"])) for t in ids], dtype=int)
    print(f"\nDeeper dial: {y.sum()} of {len(y)} tracks are {', '.join(sorted(DEEP_STYLES))} "
          "on Discogs. ROC-AUC at ranking them first:")
    rows = [idx._row_of[t] for t in ids]
    for a in ("deep", "dark", "electronic", "energy"):
        if a in idx.axis_vals:
            print(f"  {'index axis: ' + a:40s} {roc_auc_score(y, idx.axis_vals[a][rows]):.3f}")
    jd = [(idx.entries[t].get("moods") or {}).get("deep") for t in ids]
    if sum(v is not None for v in jd) > 0.9 * len(ids):
        jd = np.array([np.nan if v is None else v for v in jd])
        jd = np.where(np.isnan(jd), np.nanmean(jd), jd)
        print(f"  {'mood/theme head: deep (old axis)':40s} {roc_auc_score(y, jd):.3f}")
    for name, (vecs, words, W) in extra.items():
        T = np.stack([vecs[t] for t in ids])
        Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
        for w in ("deep", "deep house", "dub techno", "deep techno"):
            if w in words:
                auc = roc_auc_score(y, T @ Wn[words.index(w)])
                print(f"  {name + ' zero-shot: ' + repr(w + ' music'):40s} {auc:.3f}")
    for name, X in sets.items():
        auc = np.mean([evaluate(np.asarray(X, dtype=float), y[:, None], seed=s)[0]
                       for s in SEEDS])
        print(f"  {'trained on ' + name:40s} {auc:.3f}", flush=True)


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)
    idx = LibraryIndex()
    extra = load_text_audio()
    ids, styles, Y = load_labels(idx, extra)
    print(f"{len(ids)} tracks with confident Discogs styles, {len(styles)} styles "
          f"with at least {MIN_TRACKS} tracks:")
    print("  " + ", ".join(f"{s} ({Y[:, j].sum()})" for j, s in enumerate(styles)) + "\n")
    base = Y.mean(axis=0)
    order = np.argsort(-base)
    print(f"{'always guess the most common styles':34s} auc 0.500  hit@1 {np.mean(Y[:, order[0]]):.2f}  "
          f"hit@3 {np.mean(Y[:, order[:3]].any(axis=1)):.2f}")
    per_style = {}
    for name, X in feature_sets(idx, ids, extra).items():
        auc, h1, h3, aucs = evaluate_seeds(np.asarray(X, dtype=float), Y)
        per_style[name] = aucs
        print(f"{name:34s} auc {auc:.3f}  hit@1 {h1:.2f}  hit@3 {h3:.2f}", flush=True)
    for name, (vecs, words, W) in extra.items():
        T = np.stack([vecs[t] for t in ids])
        z = zero_shot(T, words, W, styles, Y)
        if z:
            auc, h1, h3, n, aucs = z
            per_style[f"{name} zero-shot"] = np.array([aucs.get(s, np.nan) for s in styles])
            print(f"{name + ' zero-shot (' + str(n) + ' styles)':34s} auc {auc:.3f}  "
                  f"hit@1 {h1:.2f}  hit@3 {h3:.2f}  (no training)", flush=True)
    dial_check(idx, ids, extra, feature_sets(idx, ids, extra))
    names = list(per_style)
    print("\nROC-AUC per style:")
    print(f"{'':20s}" + "".join(f"{n[:14]:>15s}" for n in names))
    for j, s in enumerate(styles):
        print(f"{s[:20]:20s}" + "".join(
            f"{per_style[n][j]:15.3f}" if not np.isnan(per_style[n][j]) else f"{'-':>15s}"
            for n in names))


if __name__ == "__main__":
    main()
