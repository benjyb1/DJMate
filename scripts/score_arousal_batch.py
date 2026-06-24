"""
score_arousal_batch.py — MusiCNN DEAM arousal/valence per track, cached by path.
arousal is the closest thing to "energy" Essentia offers (DEAM 1-9 scale).
"""
import sys, os, json
import numpy as np
import essentia
essentia.log.warningActive = False
essentia.log.infoActive = False
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredict2D

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")
CACHE = os.path.join(HERE, "arousal_cache.json")
MUSICNN = os.path.join(MODELS_DIR, "msd-musicnn-1.pb")
DEAM = os.path.join(MODELS_DIR, "deam-msd-musicnn-2.pb")


def main():
    arg = sys.argv[1]
    if arg == "--all":
        items = [f["filepath"] for f in json.load(open(os.path.join(HERE, "all_tracks.json")))]
    else:
        items = [d["filepath"] for d in json.load(open(arg))]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [f for f in items if f not in cache and os.path.isfile(f)]
    print(f"{len(items)} requested, {len(todo)} to compute")

    emb_model = TensorflowPredictMusiCNN(graphFilename=MUSICNN, output="model/dense/BiasAdd")
    head = TensorflowPredict2D(graphFilename=DEAM, output="model/Identity")
    for i, f in enumerate(todo, 1):
        try:
            audio = MonoLoader(filename=f, sampleRate=16000, resampleQuality=4)()
            emb = np.array(emb_model(audio))
            pred = np.array(head(emb))            # (frames, 2) = valence, arousal
            cache[f] = {"valence": float(np.mean(pred[:, 0])),
                        "arousal": float(np.mean(pred[:, 1]))}
        except Exception as exc:
            cache[f] = {"error": str(exc)}
        if i % 10 == 0 or i == len(todo):
            json.dump(cache, open(CACHE, "w")); print(f"  {i}/{len(todo)}")
    json.dump(cache, open(CACHE, "w"))
    print(f"cache holds {len(cache)} -> {CACHE}")


if __name__ == "__main__":
    main()
