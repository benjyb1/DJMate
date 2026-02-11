import os
import numpy as np
import streamlit as st

from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd

# ------------------ Human-in-the-loop corrections ------------------
if "pending_votes" not in st.session_state:
    st.session_state.pending_votes = {}

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "use_contrastive" not in st.session_state:
    st.session_state.use_contrastive = True

MUSIC_LIBRARY_ROOT = "/Users/benjyb/Desktop/Mixing"

def find_file_recursively(root_dir, target_filename):
    for root, _, files in os.walk(root_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()  # loads variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# ------------------ Data loading ------------------
@st.cache_data
def load_tracks_labels_and_feedback():
    tracks = supabase.table("tracks").select("trackid,title,artist,filepath,embedding").execute().data
    labels = supabase.table("track_labels").select("trackid,semantic_tags").execute().data
    feedback = supabase.table("tag_track_feedback").select("trackid,tag,weight_delta").execute().data

    label_map = {}
    for row in labels:
        for tag in row.get("semantic_tags", []):
            label_map.setdefault(tag, []).append(row["trackid"])

    track_map = {t["trackid"]: t for t in tracks}

    feedback_map = {}
    for row in feedback:
        feedback_map[(row["trackid"], row["tag"])] = row["weight_delta"]

    return track_map, label_map, feedback_map

# ------------------ Adaptive Centroid and Ranking Functions ------------------

def get_embedding(track):
    emb = track["embedding"]
    if isinstance(emb, str):
        emb = json.loads(emb)
    return np.array(emb, dtype=np.float32)

def compute_mean_centroid(track_map, label_map, tag):
    track_ids = label_map.get(tag, [])
    embeddings = [get_embedding(track_map[tid]) for tid in track_ids if tid in track_map]
    if not embeddings:
        return None
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid

def compute_medoid_centroid(track_map, label_map, tag):
    track_ids = label_map.get(tag, [])
    embeddings = [get_embedding(track_map[tid]) for tid in track_ids if tid in track_map]
    if not embeddings:
        return None
    sim_matrix = cosine_similarity(embeddings)
    avg_sim = (sim_matrix.sum(axis=1) - 1) / (len(track_ids) - 1)
    medoid_idx = avg_sim.argmax()
    return embeddings[medoid_idx]

from sklearn.cluster import AgglomerativeClustering

def compute_multimodal_representatives(track_map, label_map, tag, max_clusters=3):
    track_ids = label_map.get(tag, [])
    embeddings = np.array([get_embedding(track_map[tid]) for tid in track_ids if tid in track_map])
    if len(track_ids) < 4:
        return [compute_medoid_centroid(track_map, label_map, tag)]
    n_clusters = min(max_clusters, len(track_ids) // 2)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    reps = []
    for cluster_label in range(n_clusters):
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_embs = embeddings[cluster_indices]
        # pick medoid
        sim_matrix = cosine_similarity(cluster_embs)
        n = len(cluster_embs)
        if n <= 1:
            avg_sim = np.array([0.0])
        else:
            avg_sim = (sim_matrix.sum(axis=1) - 1) / (n - 1)
        medoid_idx = avg_sim.argmax()
        reps.append(cluster_embs[medoid_idx])
    return reps
def compute_density_weighted_centroid(track_map, label_map, tag, sigma=0.1):
    track_ids = label_map.get(tag, [])
    embeddings = np.array([get_embedding(track_map[tid]) for tid in track_ids if tid in track_map])
    if len(embeddings) == 0:
        return None
    sim_matrix = cosine_similarity(embeddings)
    densities = np.exp(-(1 - sim_matrix) / (2 * sigma**2)).sum(axis=1)
    weights = densities / densities.sum()
    centroid = (embeddings.T @ weights).T
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid

def rank_by_nn_voting(track_map, label_map, tag):
    labeled_ids = label_map.get(tag, [])
    labeled_embeddings = [get_embedding(track_map[tid]) for tid in labeled_ids if tid in track_map]
    scores = []
    for tid, track in track_map.items():
        emb = get_embedding(track)
        sims = [cosine_similarity([emb], [lbl_emb])[0][0] for lbl_emb in labeled_embeddings]
        avg_sim = np.mean(sims)
        scores.append((avg_sim, tid, track))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores

def rank_tracks_multimodal(track_map, representatives, tag):
    scores = []
    for tid, track in track_map.items():
        emb = get_embedding(track)
        max_sim = max([cosine_similarity([emb], [rep])[0][0] for rep in representatives])
        scores.append((max_sim, tid, track))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores

def rank_by_similarity_to_centroid(track_map, centroid):
    scores = []
    for tid, track in track_map.items():
        emb = get_embedding(track)
        sim = cosine_similarity([emb], [centroid])[0][0]
        scores.append((sim, tid, track))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores

def compute_adaptive_centroid(track_map, label_map, tag):
    track_ids = label_map.get(tag, [])
    n_tracks = len(track_ids)
    if n_tracks == 0:
        return None, "no_tracks"
    if n_tracks == 1:
        return get_embedding(track_map[track_ids[0]]), "single"
    if n_tracks <= 5:
        return None, "nn_voting"
    if n_tracks <= 15:
        return compute_medoid_centroid(track_map, label_map, tag), "medoid"
    if n_tracks <= 30:
        embeddings = np.array([get_embedding(track_map[tid]) for tid in track_ids])
        avg_sim = cosine_similarity(embeddings).mean()
        if avg_sim < 0.70:
            return compute_multimodal_representatives(track_map, label_map, tag), "multimodal"
        else:
            return compute_density_weighted_centroid(track_map, label_map, tag), "density_weighted"
    return compute_mean_centroid(track_map, label_map, tag), "mean"

def rank_tracks_adaptive(track_map, label_map, tag):
    centroid_or_reps, method = compute_adaptive_centroid(track_map, label_map, tag)
    if method == "no_tracks":
        return []
    if method == "nn_voting":
        return rank_by_nn_voting(track_map, label_map, tag)
    if method == "multimodal":
        return rank_tracks_multimodal(track_map, centroid_or_reps, tag)
    return rank_by_similarity_to_centroid(track_map, centroid_or_reps)

# ------------------ Misfit Detection ------------------
def detect_misfits(_track_map, _label_map, centroids):
    """
    Find tracks that are more similar to other tags' centroids.
    Returns: dict of {tag: [(trackid, assigned_sim, best_other_tag, best_other_sim, margin), ...]}
    """
    misfits = {}

    for tag, track_ids in _label_map.items():
        if tag not in centroids:
            continue

        tag_misfits = []

        for tid in track_ids:
            if tid not in _track_map or not _track_map[tid]["embedding"]:
                continue

            emb = _track_map[tid]["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            vec = np.array(emb, dtype=np.float32)

            # Similarity to assigned tag
            assigned_sim = cosine_similarity([centroids[tag]], [vec])[0][0]

            # Find best alternative tag
            best_other_sim = -1
            best_other_tag = None

            for other_tag, other_centroid in centroids.items():
                if other_tag == tag:
                    continue
                other_sim = cosine_similarity([other_centroid], [vec])[0][0]
                if other_sim > best_other_sim:
                    best_other_sim = other_sim
                    best_other_tag = other_tag

            # Flag as misfit if more similar to another tag
            if best_other_sim > assigned_sim:
                tag_misfits.append((
                    tid,
                    assigned_sim,
                    best_other_tag,
                    best_other_sim,
                    best_other_sim - assigned_sim  # margin
                ))

        # Sort by margin (worst first)
        tag_misfits.sort(key=lambda x: x[4], reverse=True)
        misfits[tag] = tag_misfits

    return misfits

# ------------------ Tag Quality Analysis ------------------
def analyze_tag_quality(track_map, label_map, centroids, intra_threshold=0.5, merge_threshold=0.85):
    tag_stats = []
    tags = list(centroids.keys())

    # Precompute embeddings per tag for intra-tag similarity
    tag_embeddings = {}
    for tag in tags:
        vectors = []
        for tid in label_map.get(tag, []):
            if tid in track_map and track_map[tid]["embedding"]:
                emb = track_map[tid]["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                vec = np.array(emb, dtype=np.float32)
                vectors.append(vec)
        tag_embeddings[tag] = vectors

    # Compute intra-tag similarity
    intra_similarities = {}
    for tag, vectors in tag_embeddings.items():
        if len(vectors) < 2:
            intra_similarities[tag] = np.nan
            continue
        sims = cosine_similarity(vectors)
        n = len(vectors)
        sum_sims = np.sum(sims) - n
        count = n * (n - 1)
        intra_similarities[tag] = sum_sims / count

    # Helper to get representative centroid (handles list, 2D, and None)
    def get_representative_centroid(centroid_or_list):
        if centroid_or_list is None:
            return None
        if isinstance(centroid_or_list, list) or (hasattr(centroid_or_list, "ndim") and centroid_or_list.ndim == 2):
            return centroid_or_list[0]
        return centroid_or_list

    # Compute inter-tag similarity
    inter_sim_matrix = np.zeros((len(tags), len(tags)))
    for i, tag_i in enumerate(tags):
        for j, tag_j in enumerate(tags):
            if i == j:
                inter_sim_matrix[i, j] = 1.0
            else:
                ci = get_representative_centroid(centroids[tag_i])
                cj = get_representative_centroid(centroids[tag_j])
                if ci is None or cj is None:
                    inter_sim_matrix[i, j] = np.nan
                else:
                    inter_sim_matrix[i, j] = cosine_similarity([ci], [cj])[0][0]

    for i, tag in enumerate(tags):
        n_tracks = len(label_map.get(tag, []))
        intra_sim = intra_similarities.get(tag, np.nan)
        sims = inter_sim_matrix[i]
        # Don't include self in merge candidates
        sims[i] = -1
        max_sim = np.nanmax(sims)
        top_similar_tags = [tags[idx] for idx, val in enumerate(sims) if val == max_sim and val >= merge_threshold]

        if np.isnan(intra_sim) or n_tracks < 2:
            action = "keep"
        elif intra_sim < intra_threshold:
            action = "drop"
        elif max_sim >= merge_threshold and len(top_similar_tags) > 0:
            action = "merge"
        else:
            action = "keep"

        tag_stats.append({
            "Tag": tag,
            "Num Tracks": n_tracks,
            "Intra-tag Similarity": round(intra_sim, 3) if not np.isnan(intra_sim) else None,
            "Top Similar Tags": ", ".join(top_similar_tags) if top_similar_tags else "",
            "Suggested Action": action
        })

    df = pd.DataFrame(tag_stats)
    return df

# ------------------ UI Functions ------------------
def add_pending_vote(trackid, tag, delta):
    """Add a vote to pending votes in session state - NO RERUN"""
    st.session_state.pending_votes[(trackid, tag)] = delta

def save_all_pending_votes():
    """Save all pending votes to database"""
    if not st.session_state.pending_votes:
        return 0, []

    success_count = 0
    errors = []

    for (trackid, tag), delta in st.session_state.pending_votes.items():
        try:
            supabase.table("tag_track_feedback").upsert({
                "trackid": trackid,
                "tag": tag,
                "weight_delta": delta
            }, on_conflict="trackid,tag").execute()
            success_count += 1
        except Exception as e:
            errors.append(f"Error saving vote for track {trackid}, tag '{tag}': {e}")

    return success_count, errors

def merge_tags(source_tags, target_tag):
    rows = supabase.table("track_labels").select("trackid,semantic_tags").execute().data

    updated = 0
    for row in rows:
        tags = set(row["semantic_tags"])
        if tags.intersection(source_tags):
            tags.add(target_tag)
            tags -= set(source_tags)
            supabase.table("track_labels").update({
                "semantic_tags": list(tags)
            }).eq("trackid", row["trackid"]).execute()
            updated += 1

    supabase.table("tag_operations_log").insert({
        "operation_type": "merge",
        "payload": {
            "source_tags": source_tags,
            "target_tag": target_tag,
            "affected_tracks": updated
        }
    }).execute()

# ------------------ UI ------------------
st.set_page_config(layout="wide")
st.title("Semantic Tag Centroid Diagnostics")

# Load data
track_map, label_map, feedback_map = load_tracks_labels_and_feedback()

# Convert pending votes to tuple for caching
pending_votes_tuple = tuple(st.session_state.pending_votes.items())

# Compute adaptive ranking for the selected tag
centroids = {}  # Not used anymore, but kept for interface compatibility
debug_info = {}  # Not used, kept for compatibility

# Sidebar
with st.sidebar:
    st.header("Centroid Configuration")

    # Toggle contrastive learning
    use_contrastive = st.checkbox(
        "🎯 Use Contrastive Learning",
        value=st.session_state.use_contrastive,
        help="When enabled, downvoted tracks push centroids away, upvoted unlabeled tracks pull centroids closer"
    )
    if use_contrastive != st.session_state.use_contrastive:
        st.session_state.use_contrastive = use_contrastive
        st.rerun()

    if st.session_state.use_contrastive:
        st.success("✅ Active: Votes on unlabeled tracks affect centroids")
    else:
        st.warning("⚠️ Old method: Only labeled tracks affect centroids")

    st.markdown("---")

    st.header("Tag Quality Analysis")
    st.markdown("""
    Quality metrics for each semantic tag:
    - **Intra-tag similarity**: cohesion among tracks
    - **Top similar tags**: merge candidates
    - **Suggested action**: keep, merge, or drop
    """)

    # Compute actual centroids for tag quality analysis
    actual_centroids = {}
    for t in label_map.keys():
        centroid, method = compute_adaptive_centroid(track_map, label_map, t)
        if centroid is not None:
            if method == "nn_voting":
                # For very small tags, use medoid as proxy
                centroid = compute_medoid_centroid(track_map, label_map, t)
            actual_centroids[t] = centroid

    tag_quality_df = analyze_tag_quality(track_map, label_map, actual_centroids)
    st.dataframe(tag_quality_df, use_container_width=True)

    # Ensure 'Suggested Action' exists before filtering
    if "Suggested Action" in tag_quality_df.columns:
        merge_candidates = tag_quality_df[tag_quality_df["Suggested Action"] == "merge"]["Tag"].tolist()
        drop_candidates = tag_quality_df[tag_quality_df["Suggested Action"] == "drop"]["Tag"].tolist()
    else:
        merge_candidates = []
        drop_candidates = []

    st.subheader("Tag Management")

    selected_merge_tags = st.multiselect("Select tag(s) to merge", options=merge_candidates)
    merge_target = None
    if selected_merge_tags:
        all_tags = sorted(tag_quality_df["Tag"].tolist())
        merge_target = st.selectbox("Select target tag to merge into", options=all_tags)

    if selected_merge_tags and merge_target:
        if st.button("EXECUTE MERGE"):
            merge_tags(selected_merge_tags, merge_target)
            st.cache_data.clear()
            st.success("Tags merged successfully.")
            st.rerun()

    selected_drop_tags = st.multiselect("Select tag(s) to drop", options=drop_candidates)
    if selected_drop_tags:
        st.info(f"Selected to drop: {', '.join(selected_drop_tags)} (not implemented)")

    st.markdown("---")

    st.header("Tags")
    if "selected_tag" not in st.session_state:
        st.session_state.selected_tag = None

    tags_sorted = sorted(actual_centroids.keys())
    if st.session_state.selected_tag is None:
        st.session_state.selected_tag = tags_sorted[0] if tags_sorted else None

    tag = st.selectbox("Select a semantic tag", tags_sorted,
                       index=tags_sorted.index(st.session_state.selected_tag)
                       if st.session_state.selected_tag in tags_sorted else 0)
    st.session_state.selected_tag = tag

    st.markdown("---")
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)

if tag:
    num_pending = len(st.session_state.pending_votes)
    if num_pending > 0:
        st.warning(f"⏳ {num_pending} pending vote(s) - Click 'Save All Votes' to persist to database")

    # Compute adaptive ranking
    ranked_tracks = rank_tracks_adaptive(track_map, label_map, tag)
    top_10 = ranked_tracks[:10]

    st.subheader(f"Nearest tracks to centroid: {tag}")

    # Show feedback stats
    combined_feedback = feedback_map.copy()
    combined_feedback.update(st.session_state.pending_votes)

    all_voted_tracks = set()
    for (tid, tag_key) in combined_feedback.keys():
        if tag_key == tag:
            all_voted_tracks.add(tid)
    all_voted_tracks.update(label_map.get(tag, []))

    current_feedback = [(tid, combined_feedback.get((tid, tag), 0.0)) for tid in all_voted_tracks]
    active_feedback = [f for f in current_feedback if f[1] != 0.0]

    if active_feedback:
        upvotes = [f for f in active_feedback if f[1] > 0]
        downvotes = [f for f in active_feedback if f[1] < 0]
        st.info(f"📊 Active feedback: {len(active_feedback)} tracks (↑{len(upvotes)} ↓{len(downvotes)})")

        if st.session_state.debug_mode:
            st.write("Upvoted:", [track_map.get(tid, {}).get('title', f'ID:{tid}') for tid, _ in upvotes])
            st.write("Downvoted:", [track_map.get(tid, {}).get('title', f'ID:{tid}') for tid, _ in downvotes])

    # Display tracks with color-coded similarity
    for score, tid, track in top_10:
        with st.container():
            trackid = track["trackid"]
            # Color code based on similarity
            if score >= 0.80:
                emoji = "🟢"
                quality = "Excellent match"
            elif score >= 0.65:
                emoji = "🟡"
                quality = "Good match"
            elif score >= 0.50:
                emoji = "🟠"
                quality = "Weak match"
            else:
                emoji = "🔴"
                quality = "Poor match"
            st.markdown(f"{emoji} **{track['title']}** – {track['artist']}")
            saved_delta = feedback_map.get((trackid, tag), 0.0)
            pending_delta = st.session_state.pending_votes.get((trackid, tag))
            actual_delta = pending_delta if pending_delta is not None else saved_delta
            actual_weight = max(0.0, 1.0 + actual_delta)
            if pending_delta is not None:
                st.markdown(f"Similarity: `{score:.3f}` ({quality}) | Weight delta: `{pending_delta:+.2f}` ⏳ *pending*")
            elif saved_delta != 0.0:
                st.markdown(f"Similarity: `{score:.3f}` ({quality}) | Weight delta: `{saved_delta:+.2f}` ✅ *saved*")
            else:
                st.markdown(f"Similarity: `{score:.3f}` ({quality})")
            if st.session_state.debug_mode:
                is_labeled = trackid in label_map.get(tag, [])
                st.caption(f"Debug: trackid={trackid}, labeled={is_labeled}, weight={actual_weight:.3f}")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            with col1:
                if st.button("❌ Wrong", key=f"wrong_{tag}_{trackid}"):
                    add_pending_vote(trackid, tag, -1.5)
            with col2:
                if st.button("✅ Good", key=f"good_{tag}_{trackid}"):
                    add_pending_vote(trackid, tag, 1.0)
            with col3:
                if st.button("⭐ Perfect", key=f"perfect_{tag}_{trackid}"):
                    add_pending_vote(trackid, tag, 2.0)
            with col4:
                st.markdown(" ")
            target_name = os.path.basename(track["filepath"])
            full_local_path = find_file_recursively(MUSIC_LIBRARY_ROOT, target_name)
            if full_local_path and os.path.exists(full_local_path):
                with open(full_local_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning(f"Audio file not found: {target_name}")

    # ----------------- Labeled tracks for this tag section -----------------
    st.subheader(f"Tracks labeled with '{tag}' (for positive reinforcement)")
    labeled_track_ids = label_map.get(tag, [])
    for tid in labeled_track_ids:
        track = track_map.get(tid)
        if not track:
            continue
        with st.container():
            st.markdown(f"**{track.get('title', 'Unknown')}** – {track.get('artist', 'Unknown')}")

            # Audio playback (same logic as top-10 tracks)
            target_name = os.path.basename(track["filepath"])
            full_local_path = find_file_recursively(MUSIC_LIBRARY_ROOT, target_name)
            if full_local_path and os.path.exists(full_local_path):
                with open(full_local_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning(f"Audio file not found: {target_name}")

            # Vote buttons: pressing a button overwrites previous vote and shows current pending vote
            current_vote = st.session_state.pending_votes.get((tid, tag), 0.0)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("❌ Wrong", key=f"label_wrong_{tid}"):
                    st.session_state.pending_votes[(tid, tag)] = -1.5
            with col2:
                if st.button("✅ Good", key=f"label_good_{tid}"):
                    st.session_state.pending_votes[(tid, tag)] = 1.0
            with col3:
                if st.button("⭐ Perfect", key=f"label_perfect_{tid}"):
                    st.session_state.pending_votes[(tid, tag)] = 2.0

            if st.session_state.pending_votes.get((tid, tag), 0.0) != 0:
                st.caption(f"Current pending vote: {st.session_state.pending_votes[(tid, tag)]:+.1f}")

    # Save controls - ONLY PLACE WHERE st.rerun() IS CALLED
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save All Votes", type="primary", disabled=(num_pending == 0)):
            success_count, errors = save_all_pending_votes()
            if errors:
                for error in errors:
                    st.error(error)
            if success_count > 0:
                st.success(f"✅ Saved {success_count} vote(s)!")
                st.session_state.pending_votes = {}
                st.cache_data.clear()
                st.rerun()  # ONLY RERUN HERE
    with col2:
        if num_pending > 0:
            if st.button("🗑️ Clear Pending"):
                st.session_state.pending_votes = {}
                st.rerun()  # And here for clearing