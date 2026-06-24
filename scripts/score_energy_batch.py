"""
score_energy_batch.py — compute EffNet head sub-scores for a list of audio files,
caching by filepath so nothing is recomputed.

Usage:
    .venv1/bin/python scripts/score_energy_batch.py scripts/calib_set.json
    .venv1/bin/python scripts/score_energy_batch.py --all   # every track in DB

Writes/updates scripts/energy_cache.json : { filepath: {danceability,party,aggressive,relaxed,frames} }
"""
import sys, os, json
import numpy as np
import essentia
essentia.log.warningActive = False
essentia.log.infoActive = False
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")
EFFNET_PATH = "/Users/benjyb/Desktop/OlderFiles/Models/discogs-effnet-bs64.pb"
CACHE = os.path.join(HERE, "energy_cache.json")

HEADS = {  # name -> (stem, positive label)
    "danceability": ("danceability-discogs-effnet-1", "danceable"),
    "party":        ("mood_party-discogs-effnet-1",   "party"),
    "aggressive":   ("mood_aggressive-discogs-effnet-1", "aggressive"),
    "relaxed":      ("mood_relaxed-discogs-effnet-1",  "relaxed"),
}


def pos_index(stem, label):
    meta = json.load(open(os.path.join(MODELS_DIR, f"{stem}.json")))
    return meta["classes"].index(label)


def main():
    arg = sys.argv[1]
    if arg == "--all":
        files = json.load(open(os.path.join(HERE, "all_tracks.json")))
        items = [f["filepath"] for f in files]
    else:
        items = [d["filepath"] for d in json.load(open(arg))]

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [f for f in items if f not in cache and os.path.isfile(f)]
    print(f"{len(items)} requested, {len(items)-len(todo)} cached, {len(todo)} to compute")

    emb_model = TensorflowPredictEffnetDiscogs(graphFilename=EFFNET_PATH, output="PartitionedCall:1")
    heads, pidx = {}, {}
    for n, (stem, lbl) in HEADS.items():
        heads[n] = TensorflowPredict2D(graphFilename=os.path.join(MODELS_DIR, f"{stem}.pb"), output="model/Softmax")
        pidx[n] = pos_index(stem, lbl)

    for i, f in enumerate(todo, 1):
        try:
            audio = MonoLoader(filename=f, sampleRate=16000, resampleQuality=4)()
            emb = np.array(emb_model(audio))
            rec = {"frames": int(emb.shape[0])}
            for n in HEADS:
                pred = np.array(heads[n](emb))
                rec[n] = float(np.mean(pred[:, pidx[n]]))
            cache[f] = rec
        except Exception as exc:
            cache[f] = {"error": str(exc)}
        if i % 10 == 0 or i == len(todo):
            json.dump(cache, open(CACHE, "w"))
            print(f"  {i}/{len(todo)} done")
    json.dump(cache, open(CACHE, "w"))
    print(f"cache now holds {len(cache)} files -> {CACHE}")


if __name__ == "__main__":
    main()
