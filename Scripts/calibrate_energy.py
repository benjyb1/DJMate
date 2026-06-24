"""
calibrate_energy.py — fit audio features to manual energy ratings and compare
feature sets, so we keep only what genuinely predicts your ear.

Features available per track:
  EffNet heads : danceability, party, aggressive, relaxed   (scripts/energy_cache.json)
  MusiCNN DEAM : arousal, valence                           (scripts/arousal_cache.json)
"""
import os, json
import numpy as np

HERE = os.path.dirname(__file__)
EFF = json.load(open(os.path.join(HERE, "energy_cache.json")))
ARO = json.load(open(os.path.join(HERE, "arousal_cache.json"))) if os.path.exists(os.path.join(HERE, "arousal_cache.json")) else {}
BPM = json.load(open(os.path.join(HERE, "bpm_map.json"))) if os.path.exists(os.path.join(HERE, "bpm_map.json")) else {}
CALIB = json.load(open(os.path.join(HERE, "calib_set.json")))


def feats(fp):
    e, a = EFF.get(fp), ARO.get(fp)
    if not e or "error" in e:
        return None
    d = {k: e[k] for k in ("danceability", "party", "aggressive", "relaxed")}
    if a and "error" not in a:
        d["arousal"] = a["arousal"]; d["valence"] = a["valence"]
    if fp in BPM and BPM[fp]:
        d["bpm"] = BPM[fp] / 100.0   # scale so coefficient is readable
    return d


rows = [(feats(d["filepath"]), d["energy"]) for d in CALIB]
rows = [(f, y) for f, y in rows if f is not None]
y = np.array([r[1] for r in rows], dtype=float)
print(f"calibration pairs: {len(y)}  (arousal present for "
      f"{sum('arousal' in f for f, _ in rows)})")


def fit(Xtr, ytr):
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return coef


def predict(coef, Xt):
    return np.clip(coef[0] + Xt @ coef[1:], 1, 10)


def stars(e):
    return np.clip(np.round(np.asarray(e) / 2), 1, 5).astype(int)


def evaluate(FEATS, k=5, seed=0):
    rs = [(f, yy) for (f, yy) in rows if all(k2 in f for k2 in FEATS)]
    if not rs:
        return None
    X = np.array([[f[k2] for k2 in FEATS] for f, _ in rs])
    yy = np.array([v for _, v in rs])
    idx = np.arange(len(yy)); np.random.RandomState(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    pe = np.zeros(len(yy))
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        pe[te] = predict(fit(X[tr], yy[tr]), X[te])
    mae = np.abs(pe - yy).mean()
    w1e = (np.abs(pe - yy) <= 1).mean()
    se = (stars(pe) == stars(yy)).mean()
    sw1 = (np.abs(stars(pe) - stars(yy)) <= 1).mean()
    corr = np.corrcoef(pe, yy)[0, 1]
    return dict(n=len(yy), mae=mae, w1e=w1e, star_exact=se, star_w1=sw1, corr=corr)


SETS = {
    "effnet4": ["danceability", "party", "aggressive", "relaxed"],
    "arousal_only": ["arousal"],
    "bpm_only": ["bpm"],
    "all6": ["danceability", "party", "aggressive", "relaxed", "arousal", "valence"],
    "effnet4+bpm": ["danceability", "party", "aggressive", "relaxed", "bpm"],
    "all6+bpm": ["danceability", "party", "aggressive", "relaxed", "arousal", "valence", "bpm"],
}
print(f"\n{'feature set':18s} {'n':>4} {'corr':>5} {'MAE':>5} {'±1E':>5} {'★=':>5} {'★±1':>5}")
print("-" * 50)
best = None
for name, fs in SETS.items():
    r = evaluate(fs)
    if not r:
        print(f"{name:18s}  (no data)"); continue
    print(f"{name:18s} {r['n']:>4} {r['corr']:>5.2f} {r['mae']:>5.2f} "
          f"{r['w1e']*100:>4.0f}% {r['star_exact']*100:>4.0f}% {r['star_w1']*100:>4.0f}%")
    if best is None or r["mae"] < best[1]["mae"]:
        best = (name, r, fs)

name, r, fs = best
rs = [(f, yy) for (f, yy) in rows if all(k2 in f for k2 in fs)]
X = np.array([[f[k2] for k2 in fs] for f, _ in rs])
yy = np.array([v for _, v in rs])
coef = fit(X, yy)
print(f"\nbest = {name}  (CV MAE {r['mae']:.2f}, within±1 star {r['star_w1']*100:.0f}%)")
print("  energy = %.2f + " % coef[0] + " + ".join(f"{c:.2f}*{n2}" for c, n2 in zip(coef[1:], fs)))
json.dump({"feats": fs, "coef": coef.tolist()}, open(os.path.join(HERE, "energy_model.json"), "w"), indent=2)
print("saved scripts/energy_model.json")
