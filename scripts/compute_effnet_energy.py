"""
compute_effnet_energy.py — derive a per-track energy score from the audio using
the Discogs-EffNet embedding plus Essentia classification heads.

Essentia has no arousal/DEAM head for EffNet (arousal only exists for MusiCNN /
VGGish), so "energy" here is a transparent blend of the EffNet heads that do
exist and that track intensity:

    energy = w_dance*danceable + w_party*party + w_aggr*aggressive
             + w_relax*(1 - relaxed)

Each sub-score is the mean over frames of the head's positive-class probability.
The positive class index is read from each head's .json (the class order is NOT
consistent between heads, e.g. danceability lists 'danceable' first but
mood_party lists 'party' second).

Run with the venv that has Essentia:
    /Users/benjyb/Desktop/PycharmProjects/Kings\\ Coding\\ Club/DJMate/.venv1/bin/python \\
        scripts/compute_effnet_energy.py "/Users/benjyb/Desktop/Mixing/Deep Detroit"
"""

import sys
import os
import glob
import json
import numpy as np

MODELS_DIR = "/Users/benjyb/Desktop/PycharmProjects/Kings Coding Club/DJMate/models"
EFFNET_PATH = "/Users/benjyb/Desktop/OlderFiles/Models/discogs-effnet-bs64.pb"

# head name -> (model stem, set of labels that mean "high energy", inverse?)
HEADS = {
    "danceability": ("danceability-discogs-effnet-1", {"danceable"}, False),
    "party":        ("mood_party-discogs-effnet-1",   {"party"},     False),
    "aggressive":   ("mood_aggressive-discogs-effnet-1", {"aggressive"}, False),
    "relaxed":      ("mood_relaxed-discogs-effnet-1",  {"relaxed"},   True),
}

# blend weights — relaxed contributes via (1 - relaxed_prob).
# danceability is deliberately 0: on an all-dance library it saturates at ~1.00
# for every track (measured range 0.01), so it adds no discrimination. The
# variance-carrying heads are aggressive and relaxed; party adds a little.
WEIGHTS = {"danceability": 0.00, "party": 0.40, "aggressive": 0.35, "relaxed": 0.25}


def positive_index(stem, positive_labels):
    meta = json.load(open(os.path.join(MODELS_DIR, f"{stem}.json")))
    classes = meta["classes"]
    for i, c in enumerate(classes):
        if c in positive_labels:
            return i, classes
    raise ValueError(f"{stem}: none of {positive_labels} in {classes}")


def energy_to_stars(e10):
    # Benjy's non-uniform scale: low end stricter so dub-level lands at 1 star
    if e10 <= 3.5: return 1
    if e10 <= 5.5: return 2
    if e10 <= 7.0: return 3
    if e10 <= 8.0: return 4
    return 5


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "/Users/benjyb/Desktop/Mixing/Deep Detroit"
    import essentia
    essentia.log.warningActive = False
    essentia.log.infoActive = False
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

    print(f"Loading EffNet embedding model …")
    emb_model = TensorflowPredictEffnetDiscogs(graphFilename=EFFNET_PATH, output="PartitionedCall:1")

    head_models, pos_idx = {}, {}
    for name, (stem, pos_labels, _inv) in HEADS.items():
        head_models[name] = TensorflowPredict2D(
            graphFilename=os.path.join(MODELS_DIR, f"{stem}.pb"), output="model/Softmax")
        pos_idx[name], classes = positive_index(stem, pos_labels)
        print(f"  head {name:12s} classes={classes} positive_idx={pos_idx[name]}")

    files = sorted(glob.glob(os.path.join(folder, "*.mp3")) +
                   glob.glob(os.path.join(folder, "*.flac")) +
                   glob.glob(os.path.join(folder, "*.wav")) +
                   glob.glob(os.path.join(folder, "*.aiff")) +
                   glob.glob(os.path.join(folder, "*.m4a")))
    print(f"\n{len(files)} audio files in {folder}\n")

    results = []
    for f in files:
        name = os.path.basename(f)
        try:
            audio = MonoLoader(filename=f, sampleRate=16000, resampleQuality=4)()
            emb = np.array(emb_model(audio))            # (frames, 1280)
            if emb.ndim != 2:
                raise RuntimeError(f"embedding shape {emb.shape}")
            sub = {}
            for hn in HEADS:
                pred = np.array(head_models[hn](emb))   # (frames, 2)
                p = float(np.mean(pred[:, pos_idx[hn]]))
                _, _, inv = HEADS[hn]
                sub[hn] = (1.0 - p) if inv else p
            e01 = sum(WEIGHTS[k] * sub[k] for k in HEADS)
            e10 = round(1 + 9 * e01, 2)
            stars = energy_to_stars(e10)
            results.append({"file": name, "sub": sub, "energy01": round(e01, 4),
                            "energy10": e10, "stars": stars, "emb_dim": emb.shape[1]})
            print(f"{name[:46]:46s}  dance={sub['danceability']:.2f} party={sub['party']:.2f} "
                  f"aggr={sub['aggressive']:.2f} !relax={sub['relaxed']:.2f}  "
                  f"E={e10:4.1f}/10  {'★'*stars}{'☆'*(5-stars)}")
        except Exception as exc:
            print(f"{name[:46]:46s}  ERROR: {exc}")
            results.append({"file": name, "error": str(exc)})

    safe = os.path.basename(folder.rstrip("/")).lower().replace(" ", "_")
    out = os.path.join(os.path.dirname(__file__), f"{safe}_energy.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
