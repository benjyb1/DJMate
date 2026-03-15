"""Shared embedding utilities for DJMate."""
import json
import numpy as np


def build_embedding_matrix(tracks):
    """Build a normalised embedding matrix and a parallel list of track IDs."""
    track_ids = []
    embeddings = []
    for t in tracks:
        emb = t.get("embedding")
        if emb is None:
            continue
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(emb, list):
            continue
        track_ids.append(t["trackid"])
        embeddings.append(emb)

    if not embeddings:
        return [], np.array([])

    X = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return track_ids, X
