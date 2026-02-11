import streamlit as st
st.set_page_config(layout="wide", page_title="AI Music Tagger")

import os
import numpy as np
import pandas as pd
import json

from supabase import create_client
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import normalize

# ------------------ Configuration ------------------
MUSIC_LIBRARY_ROOT = "/Users/benjyb/Desktop/Mixing"
SUPABASE_URL = 'https://cvermotfxamubejfnoje.supabase.co'
SUPABASE_KEY = 'sb_secret_1U7o2RsVAD2_5eTdBQaxkw_adLbxVBe' # Be careful exposing secrets in code!

if "pending_votes" not in st.session_state:
    st.session_state.pending_votes = {}

# ------------------ Supabase & Data ------------------
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

@st.cache_data
def load_data():
    # Fetch all data
    tracks = supabase.table("tracks").select("trackid,title,artist,filepath,embedding").execute().data
    labels = supabase.table("track_labels").select("trackid,semantic_tags").execute().data
    feedback = supabase.table("tag_track_feedback").select("trackid,tag,weight_delta").execute().data

    # Process Tracks
    track_map = {}
    valid_embeddings = []
    valid_track_ids = []

    for t in tracks:
        track_map[t["trackid"]] = t
        if t["embedding"]:
            emb = json.loads(t["embedding"]) if isinstance(t["embedding"], str) else t["embedding"]
            valid_embeddings.append(emb)
            valid_track_ids.append(t["trackid"])

    # Create Global Feature Matrix (X)
    # L2 Normalize so Euclidean distance = Cosine distance logic
    if valid_embeddings:
        X = normalize(np.array(valid_embeddings, dtype=np.float32), norm='l2')
    else:
        X = np.array([])

    # Map ID -> Matrix Index
    id_to_idx = {tid: i for i, tid in enumerate(valid_track_ids)}

    # Process Labels
    label_map = {}
    for row in labels:
        for tag in row.get("semantic_tags", []):
            label_map.setdefault(tag, []).append(row["trackid"])

    return track_map, label_map, feedback, X, valid_track_ids, id_to_idx

def find_file_recursively(root_dir, target_filename):
    for root, _, files in os.walk(root_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

# ------------------ Semi-Supervised Logic ------------------

def run_label_spreading(target_tag, track_map, label_map, feedback, X, valid_track_ids, id_to_idx):
    """
    Runs Label Spreading for a specific tag.
    Returns: list of (probability, track_id, track_data) sorted by confidence.
    """
    if X.shape[0] == 0:
        return []

    # 1. Build Label Vector (y)
    # -1 = Unlabeled (The AI fills these in)
    #  1 = Positive (Has the tag)
    #  0 = Negative (Does not have the tag)
    y = np.full(X.shape[0], -1)

    # A. Mark Positives (from Database)
    positive_ids = set(label_map.get(target_tag, []))

    # B. Mark Negatives (Implicit)
    # If a track has OTHER tags but NOT this one, assume it's a negative 0.
    # This assumption is crucial for the algorithm to have boundaries.
    all_tagged_ids = set()
    for tag, tids in label_map.items():
        all_tagged_ids.update(tids)

    # Apply Implicit Negatives (Has labels, but not THIS label)
    for tid in all_tagged_ids:
        if tid not in positive_ids and tid in id_to_idx:
            y[id_to_idx[tid]] = 0

    # Apply Positives
    for tid in positive_ids:
        if tid in id_to_idx:
            y[id_to_idx[tid]] = 1

    # C. Apply User Feedback (Overrides DB)
    # If user explicitly voted "Wrong" (-1.5), force it to 0
    # If user explicitly voted "Good", force it to 1
    for row in feedback:
        if row['tag'] == target_tag:
            tid = row['trackid']
            if tid in id_to_idx:
                if row['weight_delta'] < 0: # Downvote
                    y[id_to_idx[tid]] = 0
                elif row['weight_delta'] > 0: # Upvote
                    y[id_to_idx[tid]] = 1

    # Check if we have enough data to run
    # We need at least one positive and one negative (or unlabeled)
    if 1 not in y:
        return [] # No examples of this tag yet

    # 2. Train Model
    # kernel='knn' is robust for high dimensions. n_neighbors=7 is a sweet spot for music.
    # alpha=0.2 means "Keep 80% of original labels, let 20% change if neighbors disagree deeply"
    model = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
    model.fit(X, y)

    # 3. Get Probabilities
    # output_distributions_ gives [prob_class_0, prob_class_1]
    probs = model.predict_proba(X)[:, 1]

    # 4. Package Results
    results = []
    for idx, prob in enumerate(probs):
        tid = valid_track_ids[idx]

        # Filter: Only show UNTAGGED tracks (where y was -1 or implicit 0)
        # We don't need AI to tell us what we already tagged as Positive
        if tid not in positive_ids:
            results.append((prob, tid, track_map[tid]))

    # Sort by probability (High confidence first)
    results.sort(key=lambda x: x[0], reverse=True)

    return results

# ------------------ UI Helpers ------------------
def add_pending_vote(trackid, tag, delta):
    st.session_state.pending_votes[(trackid, tag)] = delta

def save_votes():
    count = 0
    for (tid, tag), delta in st.session_state.pending_votes.items():
        supabase.table("tag_track_feedback").upsert({
            "trackid": tid, "tag": tag, "weight_delta": delta
        }, on_conflict="trackid,tag").execute()

        # OPTIONAL: Auto-tag in main table if vote is "Perfect"
        if delta >= 2.0:
            current_tags = supabase.table("track_labels").select("semantic_tags").eq("trackid", tid).execute().data
            if current_tags:
                tags = set(current_tags[0]['semantic_tags'] or [])
                tags.add(tag)
                supabase.table("track_labels").update({"semantic_tags": list(tags)}).eq("trackid", tid).execute()

        count += 1
    st.session_state.pending_votes = {}
    st.cache_data.clear()
    st.success(f"Saved {count} votes & retrained model!")
    st.rerun()


# Load Data

track_map, label_map, feedback, X, valid_track_ids, id_to_idx = load_data()

# ------------------ Remove Duplicate Track Names ------------------
seen_titles = set()
filtered_indices = []
filtered_valid_ids = []
for idx, tid in enumerate(valid_track_ids):
    title_raw = track_map[tid].get('title')
    if not title_raw:
        continue  # Skip tracks with no title
    title = title_raw.strip().lower()
    if title not in seen_titles:
        seen_titles.add(title)
        filtered_indices.append(idx)
        filtered_valid_ids.append(tid)

# Filter X and valid_track_ids
if len(filtered_indices) < len(valid_track_ids):
    X = X[filtered_indices]
    valid_track_ids = filtered_valid_ids
    id_to_idx = {tid: i for i, tid in enumerate(valid_track_ids)}

st.title("🤖 Semi-Supervised Auto-Tagger")

with st.sidebar:
    st.header("Control Panel")
    all_tags = sorted(list(label_map.keys()))
    selected_tag = st.selectbox("Select Tag to Expand", all_tags)

    st.markdown("---")
    st.info(f"📚 Knowledge Base:\n- Total Tracks: {len(track_map)}\n- Labeled with '{selected_tag}': {len(label_map.get(selected_tag, []))}")

    if len(st.session_state.pending_votes) > 0:
        st.warning(f"{len(st.session_state.pending_votes)} unsaved changes")
        if st.button("💾 Save & Retrain", type="primary"):
            save_votes()

# --- Main Content ---

if selected_tag:
    # RUN AI
    # In a real app, you might want a button to trigger this to save compute,
    # but for 600 tracks, this is near-instant.
    predictions = run_label_spreading(selected_tag, track_map, label_map, feedback, X, valid_track_ids, id_to_idx)

    if not predictions:
        st.warning("Not enough data to run AI. Tag at least 1 song with this tag (and 1 song with a different tag).")
    else:
        # Split into High Confidence (Almost sure) and Uncertain (Needs human help)
        high_conf = [p for p in predictions if p[0] > 0.85]
        uncertain = [p for p in predictions if 0.4 < p[0] < 0.6]

        # --- TAB VIEW ---
        tab1, tab2, tab3 = st.tabs(["🚀 High Confidence (Quick Approve)", "🤔 Uncertain (Active Learning)", "✅ Already Tagged"])

        # 1. HIGH CONFIDENCE
        with tab1:
            st.markdown(f"**AI found {len(high_conf)} tracks that are highly likely to be '{selected_tag}'**")
            st.markdown("These are the songs deeply embedded in the cluster of existing tags.")

            for prob, tid, track in high_conf[:10]:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([4, 2, 2])
                    with col1:
                        st.markdown(f"**{track['title']}**")
                        st.caption(f"{track['artist']} • Confidence: `{prob:.1%}`")

                        # Audio Player
                        fname = os.path.basename(track["filepath"])
                        local_path = find_file_recursively(MUSIC_LIBRARY_ROOT, fname)
                        if local_path:
                            st.audio(local_path)

                    with col2:
                        # Logic to handle UI state for pending votes
                        vote = st.session_state.pending_votes.get((tid, selected_tag))
                        if vote == 1.0:
                            st.success("Accepted")
                        elif vote == -1.5:
                            st.error("Rejected")
                        else:
                            if st.button("Accept", key=f"acc_{tid}"):
                                add_pending_vote(tid, selected_tag, 1.0)
                                st.rerun()

                    with col3:
                        if st.button("Reject", key=f"rej_{tid}"):
                            add_pending_vote(tid, selected_tag, -1.5)
                            st.rerun()

        # 2. UNCERTAIN / EDGE CASES
        with tab2:
            st.markdown(f"**AI is unsure about these {len(uncertain)} tracks.**")
            st.markdown("Tagging these provides the **most value** to the AI (Active Learning).")

            for prob, tid, track in uncertain[:10]:
                with st.container(border=True):
                    st.markdown(f"❓ **{track['title']}** - {track['artist']}")
                    st.caption(f"Confidence: `{prob:.1%}` (The AI is confused)")

                    fname = os.path.basename(track["filepath"])
                    local_path = find_file_recursively(MUSIC_LIBRARY_ROOT, fname)
                    if local_path:
                        st.audio(local_path)

                    c1, c2, c3 = st.columns(3)
                    if st.button("Not this vibe", key=f"u_no_{tid}"):
                        add_pending_vote(tid, selected_tag, -1.5)
                        st.rerun()
                    if st.button("Is this vibe", key=f"u_yes_{tid}"):
                        add_pending_vote(tid, selected_tag, 1.0)
                        st.rerun()
                    if st.button("Skip", key=f"u_skip_{tid}"):
                        pass

        # 3. EXISTING
        with tab3:
            st.dataframe(pd.DataFrame(
                [track_map[tid] for tid in label_map.get(selected_tag, [])],
                columns=["title", "artist", "filepath"]
            ))