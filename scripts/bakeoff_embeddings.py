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


def load_labels(idx):
    raw = json.loads((paths.data_dir() / "discogs.json").read_text())
    rows = [(tid, v["styles"]) for tid, v in raw.items()
            if v.get("status") == "ok" and v.get("confidence", 0) >= MIN_CONFIDENCE
            and v.get("styles") and tid in idx._row_of]
    counts = Counter(s for _, st in rows for s in st)
    styles = sorted(s for s, n in counts.items() if n >= MIN_TRACKS)
    rows = [(tid, [s for s in st if s in styles]) for tid, st in rows]
    rows = [(tid, st) for tid, st in rows if st]
    Y = np.array([[s in st for s in styles] for _, st in rows], dtype=int)
    return [tid for tid, _ in rows], styles, Y


def feature_sets(idx, ids):
    rows = [idx._row_of[t] for t in ids]
    ents = [idx.entries[t] for t in ids]
    X = np.stack([idx._emb[t] for t in ids]).astype(np.float32)
    half = X.shape[1] // 2
    sets = {
        "EffNet mean+std (current)": X,
        "EffNet mean only": X[:, :half],
    }
    if all(t in idx._intro and t in idx._outro for t in ids):
        sets["EffNet mid+intro+outro"] = np.hstack([
            X[:, :half],
            np.stack([idx._intro[t] for t in ids]),
            np.stack([idx._outro[t] for t in ids])])
    heads = sorted(ents[0]["heads"])
    sets["7 mood/genre heads only"] = np.array([[e["heads"][h] for h in heads] for e in ents])
    if all(e.get("moods") for e in ents):
        mk = sorted(ents[0]["moods"])
        M = np.array([[e["moods"][k] for k in mk] for e in ents])
        sets["Jamendo mood/theme (56)"] = M
        sets["EffNet mean + mood/theme"] = np.hstack([X[:, :half], M * 10])
    return sets


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
            pca = PCA(128, random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        for j in range(Y.shape[1]):
            if Y[tr, j].min() == Y[tr, j].max():
                continue
            m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
            m.fit(Xtr, Y[tr, j])
            P[te, j] = m.predict_proba(Xte)[:, 1]
    aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])
            if 0 < Y[:, j].sum() < len(Y)]
    order = np.argsort(-P, axis=1)
    hit1 = np.mean([Y[i, order[i, 0]] for i in range(len(Y))])
    hit3 = np.mean([Y[i, order[i, :3]].any() for i in range(len(Y))])
    return float(np.mean(aucs)), float(hit1), float(hit3)


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)
    idx = LibraryIndex()
    ids, styles, Y = load_labels(idx)
    print(f"{len(ids)} tracks with confident Discogs styles, {len(styles)} styles "
          f"with at least {MIN_TRACKS} tracks:")
    print("  " + ", ".join(f"{s} ({Y[:, j].sum()})" for j, s in enumerate(styles)) + "\n")
    base = Y.mean(axis=0)
    order = np.argsort(-base)
    print(f"{'always guess the most common styles':34s} auc 0.500  hit@1 {np.mean(Y[:, order[0]]):.2f}  "
          f"hit@3 {np.mean(Y[:, order[:3]].any(axis=1)):.2f}")
    for name, X in feature_sets(idx, ids).items():
        auc, h1, h3 = evaluate(np.asarray(X, dtype=float), Y)
        print(f"{name:34s} auc {auc:.3f}  hit@1 {h1:.2f}  hit@3 {h3:.2f}", flush=True)


if __name__ == "__main__":
    main()
