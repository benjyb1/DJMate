"""
auto_tagger.py
──────────────
Batch auto-tagger: uses the same ML suggestion logic as Tagger.py's AI Brain
to populate track_labels for every track that has no existing label entry.

Algorithm (identical to get_ai_suggestions in Tagger.py):
  1. Load all embeddings into memory (normalised)
  2. For each untagged track, find its 10 most similar tracks by cosine similarity
  3. Aggregate tags/vibes/energy from those neighbours, weighted by similarity
  4. Write to track_labels with tag_source = 'auto'

Safety:
  - Only writes if similarity confidence is high enough (MIN_SIMILARITY threshold)
  - Never overwrites rows that already have tag_source = 'manual' or 'auto_reviewed'
  - Dry-run mode (--dry-run flag) prints what would be written without touching DB

Usage:
  python Scripts/Backend/auto_tagger.py               # tag all untagged
  python Scripts/Backend/auto_tagger.py --dry-run     # preview only
  python Scripts/Backend/auto_tagger.py --min-sim 0.8 # stricter threshold
"""

import os
import sys
import json
import argparse
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

# Minimum cosine similarity of the closest neighbour required before we
# trust the aggregated tags. Tracks with no neighbour above this threshold
# are skipped (not enough signal to auto-tag reliably).
DEFAULT_MIN_SIMILARITY = 0.70

# Number of neighbours to aggregate tags from (same as Tagger.py AI Brain)
N_NEIGHBOURS = 10

# Minimum aggregated weight for a tag/vibe to be included in the output.
# Prevents very weak signals from being written as tags.
MIN_TAG_WEIGHT = 0.5


# ── Supabase setup ────────────────────────────────────────────────────────────

def get_supabase():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌  SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_tracks(supabase):
    """Load all tracks that have embeddings."""
    print("Loading tracks with embeddings …")
    resp = supabase.table("tracks").select("trackid, title, artist, bpm, key, filepath, embedding") \
        .not_.is_("embedding", "null").execute()
    tracks = resp.data or []
    print(f"  {len(tracks)} tracks with embeddings")
    return tracks


def load_existing_labels(supabase):
    """Load all existing track_labels rows."""
    print("Loading existing labels …")
    resp = supabase.table("track_labels") \
        .select("trackid, semantic_tags, vibe, energy, tag_source").execute()
    by_id = {}
    for row in (resp.data or []):
        by_id[row["trackid"]] = row
    print(f"  {len(by_id)} tracks already labelled")
    return by_id


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


# ── ML suggestion (identical logic to Tagger.py get_ai_suggestions) ──────────

def get_suggestions(target_trackid, track_ids, X, label_memory, vibe_memory, energy_memory):
    """
    Given a target track, find its N_NEIGHBOURS nearest neighbours by
    cosine similarity and aggregate their tags/vibes/energy.

    Returns (genre_scores, vibe_scores, predicted_energy, max_similarity)
    where *_scores are dicts of {tag: confidence_0_to_1}.
    Returns ([], [], None, 0.0) if there is not enough signal.
    """
    if target_trackid not in track_ids:
        return [], [], None, 0.0

    idx = track_ids.index(target_trackid)
    target_vec = X[idx].reshape(1, -1)
    sims = cosine_similarity(target_vec, X)[0]

    # Top N neighbours (excluding self)
    top_indices = np.argsort(sims)[::-1][1:N_NEIGHBOURS + 1]
    max_sim = float(sims[top_indices[0]]) if len(top_indices) else 0.0

    genre_scores = {}
    vibe_scores = {}
    total_weight = 0.0
    energy_weighted_sum = 0.0
    energy_weight_total = 0.0

    for i in top_indices:
        tid = track_ids[i]
        weight = float(sims[i])
        tags = label_memory.get(tid, [])
        for tag in tags:
            genre_scores[tag] = genre_scores.get(tag, 0.0) + weight
        vibes = vibe_memory.get(tid, [])
        for vibe in vibes:
            vibe_scores[vibe] = vibe_scores.get(vibe, 0.0) + weight
        if tid in energy_memory:
            energy_weighted_sum += energy_memory[tid] * weight
            energy_weight_total += weight
        if tags or vibes:
            total_weight += weight

    if total_weight == 0:
        return [], [], None, max_sim

    def format_scores(scores_dict):
        # Normalise by N_NEIGHBOURS so score ≈ fraction of neighbours with that tag
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return [
            (tag, min(0.99, score / float(N_NEIGHBOURS)))
            for tag, score in sorted_items
            if score >= MIN_TAG_WEIGHT
        ]

    predicted_energy = energy_weighted_sum / energy_weight_total if energy_weight_total > 0 else None
    return format_scores(genre_scores), format_scores(vibe_scores), predicted_energy, max_sim


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-tag untagged tracks using ML similarity.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching the DB.")
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIMILARITY,
                        help=f"Minimum cosine similarity threshold (default {DEFAULT_MIN_SIMILARITY}).")
    parser.add_argument("--overwrite-auto", action="store_true",
                        help="Re-run auto-tagging for tracks already tagged as 'auto'.")
    args = parser.parse_args()

    supabase = get_supabase()

    # ── Load data ─────────────────────────────────────────────────────────────
    all_tracks = load_all_tracks(supabase)
    existing_labels = load_existing_labels(supabase)

    track_ids, X = build_embedding_matrix(all_tracks)
    if not track_ids:
        print("❌  No tracks with valid embeddings found. Exiting.")
        sys.exit(1)

    # Build label lookup dicts (only from manually/auto_reviewed tagged tracks
    # — we don't want auto tags to bootstrap more auto tags in a noisy cascade)
    label_memory = {}
    vibe_memory = {}
    energy_memory = {}
    trusted_sources = {"manual", "auto_reviewed"}

    for tid, row in existing_labels.items():
        source = row.get("tag_source", "manual")
        if source not in trusted_sources:
            continue
        tags = row.get("semantic_tags") or []
        vibes = row.get("vibe") or []
        energy = row.get("energy")
        if tags:
            label_memory[tid] = tags if isinstance(tags, list) else []
        if vibes:
            vibe_memory[tid] = vibes if isinstance(vibes, list) else []
        if energy is not None:
            energy_memory[tid] = float(energy)

    trusted_count = len(label_memory)
    print(f"\n{trusted_count} manually-tagged tracks available as signal source.")
    if trusted_count < 5:
        print("⚠️  Very few manually-tagged tracks — auto-tag quality will be low.")
        print("    Consider tagging more tracks manually in Tagger.py first.")

    # ── Identify tracks to tag ────────────────────────────────────────────────
    # tag_source meanings:
    #   None / NULL  — row exists but has no tags yet (auto-tagger should claim it)
    #   'auto'       — previously auto-tagged (skip unless --overwrite-auto)
    #   'manual'     — human-tagged (never overwrite)
    #   'auto_reviewed' — human confirmed auto tags (never overwrite)
    tracks_by_id = {t["trackid"]: t for t in all_tracks}
    to_tag = []
    skipped_manual = 0
    skipped_already_auto = 0

    for tid in track_ids:
        existing = existing_labels.get(tid)
        if existing:
            source = existing.get("tag_source")  # may be None (NULL in DB)
            if source in ("manual", "auto_reviewed"):
                skipped_manual += 1
                continue
            if source == "auto" and not args.overwrite_auto:
                skipped_already_auto += 1
                continue
            # source is None (NULL) — needs tagging, fall through to to_tag
        to_tag.append(tid)

    print(f"\nTracks to auto-tag: {len(to_tag)}")
    print(f"Skipped (manual/reviewed): {skipped_manual}")
    print(f"Skipped (already auto, use --overwrite-auto to redo): {skipped_already_auto}")

    if not to_tag:
        print("\n✅  Nothing to tag. All done.")
        return

    if args.dry_run:
        print("\n[DRY RUN — no writes to DB]\n")

    # ── Auto-tag loop ─────────────────────────────────────────────────────────
    tagged = 0
    skipped_low_sim = 0
    skipped_no_signal = 0

    for i, tid in enumerate(to_tag):
        track = tracks_by_id.get(tid, {})
        title = track.get("title", tid)
        artist = track.get("artist", "")

        genre_scores, vibe_scores, predicted_energy, max_sim = get_suggestions(
            tid, track_ids, X, label_memory, vibe_memory, energy_memory
        )

        progress = f"[{i+1}/{len(to_tag)}]"

        if max_sim < args.min_sim:
            skipped_low_sim += 1
            print(f"  {progress} SKIP (low sim {max_sim:.2f}) — {title} — {artist}")
            continue

        if not genre_scores:
            skipped_no_signal += 1
            print(f"  {progress} SKIP (no tag signal)  — {title} — {artist}")
            continue

        final_tags = [tag for tag, _ in genre_scores]
        final_vibes = [vibe for vibe, _ in vibe_scores]
        energy = round(predicted_energy) if predicted_energy is not None else None

        print(f"  {progress} TAG  (sim {max_sim:.2f}) — {title} — {artist}")
        print(f"         tags:  {final_tags}")
        print(f"         vibes: {final_vibes}  energy: {energy}")

        if not args.dry_run:
            payload = {
                "trackid": tid,
                "semantic_tags": final_tags,
                "vibe": final_vibes,
                "energy": energy,
                "tag_source": "auto",
            }
            existing = existing_labels.get(tid)
            try:
                if existing:
                    supabase.table("track_labels").update(payload).eq("trackid", tid).execute()
                else:
                    supabase.table("track_labels").insert(payload).execute()
                tagged += 1
            except Exception as e:
                print(f"         ⚠️  DB write failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if args.dry_run:
        print("DRY RUN complete — no changes written.")
    else:
        print(f"Auto-tagging complete.")
        print(f"  Tagged:            {tagged}")
    print(f"  Skipped (low sim): {skipped_low_sim}  (threshold: {args.min_sim})")
    print(f"  Skipped (no sig):  {skipped_no_signal}")
    print("=" * 50)


if __name__ == "__main__":
    main()
