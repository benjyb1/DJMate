from supabase import create_client
import numpy as np
import faiss

from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()  # loads variables from .env
load_dotenv()  # loads variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
import numpy as np
import faiss
import re


# --- HELPER: Parse Supabase string to Numpy ---
def parse_embedding(embedding_data):
    if isinstance(embedding_data, str):
        # Remove brackets and split by comma
        return np.fromstring(embedding_data.strip('[]'), sep=',', dtype=np.float32)
    return np.array(embedding_data, dtype=np.float32)


# --- HELPER: Camelot Key Compatibility ---
def get_compatible_camelot_keys(key):
    """
    Implements harmonic mixing rules:
    1. Same Key (e.g., 8A -> 8A)
    2. Relative Major/Minor (e.g., 8A -> 8B)
    3. Perfect Fifth/Fourth (e.g., 8A -> 7A or 9A)
    """
    if not key or not re.match(r"^\d{1,2}[AB]$", key):
        return [key]

    val = int(key[:-1])
    letter = key[-1]

    # Adjacent numbers (12 wraps to 1)
    prev_val = 12 if val == 1 else val - 1
    next_val = 1 if val == 12 else val + 1
    other_letter = "B" if letter == "A" else "A"

    return [
        f"{val}{letter}",  # Same
        f"{val}{other_letter}",  # Toggle Major/Minor
        f"{prev_val}{letter}",  # Down a fifth
        f"{next_val}{letter}"  # Up a fifth
    ]


# --- Updated load_tracks with parsing ---
def load_tracks():
    response = supabase.table("tracks").select("trackid, bpm, key, embedding").execute()
    rows = response.data

    track_ids = []
    embeddings_list = []
    track_meta = {}

    for r in rows:
        # FIX: Parse string to numpy array
        vec = parse_embedding(r["embedding"])

        # Check if vector is valid (not empty)
        if vec.size > 0:
            embeddings_list.append(vec)
            track_ids.append(r["trackid"])
            track_meta[r["trackid"]] = {
                "bpm": r["bpm"],
                "key": r["key"],
            }

    return np.array(embeddings_list, dtype="float32"), track_ids, track_meta
# --- FAISS index build ---
def build_faiss_index(embeddings: np.ndarray):
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)
    return index

# --- Minimal key compatibility function ---
def key_compatible(key_a, key_b):
    compatible_keys = get_compatible_camelot_keys(key_a)
    return key_b in compatible_keys

def compute_confidence(faiss_score, seed_bpm, candidate_bpm, seed_key, candidate_key, bpm_tolerance=8.0):
    # Base: embedding similarity, weakened influence
    score = faiss_score * 0.7

    # Soft BPM modifier
    bpm_factor = 1.0
    if seed_bpm and candidate_bpm:
        diff = abs(candidate_bpm - seed_bpm)
        bpm_factor = max(0.0, 1 - diff / bpm_tolerance)

    # Soft key modifier with increased weight
    key_factor = 0.0
    if seed_key and candidate_key:
        if candidate_key == seed_key:
            key_factor = 1.5
        elif candidate_key in get_compatible_camelot_keys(seed_key):
            key_factor = 1.2
        else:
            key_factor = 0.3  # allow weak key matches instead of 0

    # Optional small bonus for harmonic match even if embedding is weak
    harmonic_bonus = 0.05 if key_factor > 0 else 0.0

    # Final confidence
    confidence_score = score * bpm_factor * key_factor + harmonic_bonus
    return confidence_score

# --- FAISS similarity function ---
def faiss_similar_tracks(
    seed_trackid: int,
    k: int = 100,
    bpm_tolerance: float = 8.0,
    top_n: int = 5,
):
    seed_faiss_id = trackid_to_faiss_id[seed_trackid]
    seed_vec = embeddings[seed_faiss_id].reshape(1, -1)

    D, I = index.search(seed_vec, k)

    seed_meta = track_meta[seed_trackid]
    seed_bpm = seed_meta["bpm"]
    seed_key = seed_meta["key"]

    results = []

    for faiss_id, score in zip(I[0], D[0]):
        if faiss_id < 0:
            continue

        tid = faiss_id_to_trackid[faiss_id]
        # Skip the seed track itself
        if tid == seed_trackid:
            continue

        meta = track_meta[tid]
        candidate_bpm = meta["bpm"]
        candidate_key = meta["key"]

        confidence = compute_confidence(
            float(score),
            seed_bpm,
            candidate_bpm,
            seed_key,
            candidate_key,
            bpm_tolerance=bpm_tolerance,
        )

        results.append({
            "trackid": tid,
            "score": confidence,
        })

    # Sort by descending confidence score
    results.sort(key=lambda x: x["score"], reverse=True)
    # Return only the top N tracks
    return results[:top_n]

# --- New function to explain similarity ---
def explain_track_similarity(seed_trackid, candidate_trackid, bpm_tolerance=8.0):
    seed_id = trackid_to_faiss_id[seed_trackid]
    candidate_id = trackid_to_faiss_id[candidate_trackid]

    seed_vec = embeddings[seed_id].reshape(1, -1)
    candidate_vec = embeddings[candidate_id].reshape(1, -1)

    # Compute FAISS similarity score (cosine similarity)
    faiss_score = float(np.dot(seed_vec, candidate_vec.T))

    seed_meta = track_meta[seed_trackid]
    candidate_meta = track_meta[candidate_trackid]

    seed_bpm = seed_meta["bpm"]
    candidate_bpm = candidate_meta["bpm"]
    seed_key = seed_meta["key"]
    candidate_key = candidate_meta["key"]

    # BPM difference and factor
    bpm_diff = abs(candidate_bpm - seed_bpm) if seed_bpm and candidate_bpm else None
    bpm_factor = max(0.0, 1 - bpm_diff / bpm_tolerance) if bpm_diff is not None else None

    # Key compatibility and factor
    if seed_key and candidate_key:
        if candidate_key == seed_key:
            key_factor = 1.0
        elif candidate_key in get_compatible_camelot_keys(seed_key):
            key_factor = 0.8
        else:
            key_factor = 0.2
        key_compatible_flag = candidate_key in get_compatible_camelot_keys(seed_key)
    else:
        key_factor = None
        key_compatible_flag = None

    # Harmonic bonus
    harmonic_bonus = 0.05 if key_factor and key_factor > 0 else 0.0

    # Final confidence
    confidence_score = compute_confidence(
        faiss_score,
        seed_bpm,
        candidate_bpm,
        seed_key,
        candidate_key,
        bpm_tolerance=bpm_tolerance,
    )

    explanation = {
        "seed_trackid": seed_trackid,
        "candidate_trackid": candidate_trackid,
        "faiss_score": faiss_score,
        "bpm_diff": bpm_diff,
        "bpm_factor": bpm_factor,
        "seed_key": seed_key,
        "candidate_key": candidate_key,
        "key_compatible": key_compatible_flag,
        "key_factor": key_factor,
        "harmonic_bonus": harmonic_bonus,
        "final_confidence": confidence_score,
    }

    return explanation

# --- Startup: load tracks and build index ---
embeddings, track_ids, track_meta = load_tracks()
index = build_faiss_index(embeddings)
faiss_id_to_trackid = {i: track_ids[i] for i in range(len(track_ids))}
trackid_to_faiss_id = {track_ids[i]: i for i in range(len(track_ids))}

# --- Test the function ---
if __name__ == "__main__":
    test_track_id = track_ids[7]  # pick a track different from the first
    results = faiss_similar_tracks(test_track_id, k=50)

    print(f"Tracks similar to {test_track_id}:")
    for r in results:
        tid = r["trackid"]
        meta = track_meta[tid]
        print(f"TrackID: {tid}, Score: {r['score']}, BPM: {meta['bpm']}, Key: {meta['key']}")
'''
    # Test explain_track_similarity function
    if len(track_ids) > 8:
        candidate_track_id = track_ids[9]
        explanation = explain_track_similarity(test_track_id, candidate_track_id)
        print("\nExplanation of similarity between tracks:")
        for k, v in explanation.items():
            print(f"{k}: {v}")'''