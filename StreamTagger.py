import streamlit as st
from supabase import create_client, Client
import random
import os
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# Supabase setup
# ======================

from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()  # loads variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# ML "Brain" Functions
# ======================

@st.cache_data(ttl=300)
def load_ml_context():
    tracks_resp = supabase.table("tracks").select("trackid,embedding").not_.is_("embedding", "null").execute()
    labels_resp = supabase.table("track_labels").select("trackid,semantic_tags,vibe,energy").execute()

    label_memory = {}
    vibe_memory = {}
    energy_memory = {}

    for row in labels_resp.data:
        tid = row['trackid']
        if row.get('semantic_tags'): label_memory[tid] = row['semantic_tags']
        if row.get('vibe'): vibe_memory[tid] = row['vibe']
        if row.get('energy') is not None: energy_memory[tid] = row['energy']

    track_ids = []
    embeddings = []

    for row in tracks_resp.data:
        if row['embedding']:
            emb = json.loads(row['embedding']) if isinstance(row['embedding'], str) else row['embedding']
            track_ids.append(row['trackid'])
            embeddings.append(emb)

    if embeddings:
        X = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
    else:
        X = np.array([])

    return track_ids, X, label_memory, vibe_memory, energy_memory

def get_ai_suggestions(current_track_id, track_ids, X, label_memory, vibe_memory, energy_memory):
    if current_track_id not in track_ids:
        return [], [], None

    idx = track_ids.index(current_track_id)
    target_vec = X[idx].reshape(1, -1)
    sims = cosine_similarity(target_vec, X)[0]
    top_indices = np.argsort(sims)[::-1][1:11]

    genre_scores = {}
    vibe_scores = {}
    total_weight = 0
    energy_weighted_sum = 0.0
    energy_weight_total = 0.0

    for i in top_indices:
        tid = track_ids[i]
        weight = sims[i]
        tags = label_memory.get(tid, [])
        for tag in tags:
            genre_scores[tag] = genre_scores.get(tag, 0) + weight
        vibes = vibe_memory.get(tid, [])
        for vibe in vibes:
            vibe_scores[vibe] = vibe_scores.get(vibe, 0) + weight
        if tid in energy_memory:
            energy_weighted_sum += energy_memory[tid] * weight
            energy_weight_total += weight
        if tags or vibes:
            total_weight += weight

    if total_weight == 0:
        return [], [], None

    def format_scores(scores_dict):
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return [(tag, min(0.99, score / 4.0)) for tag, score in sorted_items if score > 0.5]

    predicted_energy = None
    if energy_weight_total > 0:
        predicted_energy = energy_weighted_sum / energy_weight_total

    return format_scores(genre_scores), format_scores(vibe_scores), predicted_energy

# ======================
# Helper functions
# ======================

@st.cache_data(ttl=60)
def get_existing_tags():
    labels_resp = supabase.table("track_labels").select("semantic_tags").execute()
    all_tags = set()
    if labels_resp.data:
        for row in labels_resp.data:
            if row.get("semantic_tags"): all_tags.update(row["semantic_tags"])
    return sorted(list(all_tags))

@st.cache_data(ttl=60)
def get_existing_vibes():
    labels_resp = supabase.table("track_labels").select("vibe").execute()
    all_vibes = set()
    if labels_resp.data:
        for row in labels_resp.data:
            if row.get("vibe"): all_vibes.update(row["vibe"])
    return sorted(list(all_vibes))

def get_unlabeled_track():
    tracks_resp = supabase.table("tracks").select("trackid,title,artist,filepath,bpm,key").execute()
    if not tracks_resp.data: return None

    labels_resp = supabase.table("track_labels").select("trackid,semantic_tags").execute()
    tagged_ids = {row["trackid"] for row in labels_resp.data if row.get("semantic_tags")}

    unlabeled_tracks = [t for t in tracks_resp.data if t["trackid"] not in tagged_ids]
    if not unlabeled_tracks: return None

    if "seen_track_ids" not in st.session_state: st.session_state.seen_track_ids = set()
    unseen_tracks = [t for t in unlabeled_tracks if t["trackid"] not in st.session_state.seen_track_ids]

    if not unseen_tracks:
        st.session_state.seen_track_ids = set()
        unseen_tracks = unlabeled_tracks

    selected_track = random.choice(unseen_tracks)
    st.session_state.seen_track_ids.add(selected_track["trackid"])
    return selected_track

def save_to_db(trackid, semantic_tags_list, energy, vibe_list):
    existing = supabase.table("track_labels").select("trackid").eq("trackid", trackid).execute()
    payload = {
        "trackid": trackid,
        "semantic_tags": semantic_tags_list,
        "energy": energy,
        "vibe": vibe_list,
    }
    if existing.data:
        supabase.table("track_labels").update(payload).eq("trackid", trackid).execute()
    else:
        supabase.table("track_labels").insert(payload).execute()

    get_existing_tags.clear()
    get_existing_vibes.clear()
    load_ml_context.clear()

# ======================
# Main UI
# ======================

st.set_page_config(layout="wide", page_title="AI DJ Training Ground")
st.title("🎧 AI DJ Training Ground")

# Initialize Session State
if "selected_tags" not in st.session_state: st.session_state.selected_tags = []
if "selected_vibes" not in st.session_state: st.session_state.selected_vibes = []
if "reload_track" not in st.session_state: st.session_state.reload_track = False

# --- FIX: Clear input values BEFORE widgets are instantiated ---
if st.session_state.reload_track:
    st.session_state["new_genre_input"] = ""
    st.session_state["new_vibe_input"] = ""

# Load Track
if "current_track" not in st.session_state or st.session_state.reload_track:
    st.session_state.current_track = get_unlabeled_track()
    st.session_state.reload_track = False
    st.session_state.selected_tags = []
    st.session_state.selected_vibes = []

current_track = st.session_state.current_track

if not current_track:
    st.success("🎉 All tracks have been labeled!")
    st.balloons()
    st.stop()

track_ids, X, label_mem, vibe_mem, energy_mem = load_ml_context()
ai_genres, ai_vibes, ai_energy = get_ai_suggestions(
    current_track['trackid'], track_ids, X, label_mem, vibe_mem, energy_mem
)

col_left, col_center, col_right = st.columns([1, 2, 1])

# --- LEFT: Audio & Metadata ---
with col_left:
    st.subheader(f"{current_track['title']}")
    st.caption(f"{current_track['artist']}")

    filepath = current_track["filepath"]
    if os.path.exists(filepath):
        st.audio(filepath, format="audio/mp3")
    else:
        st.error("File not found")

    st.info(f"BPM: **{current_track['bpm']}** \nKey: **{current_track['key']}**")
    st.markdown("---")

    total_count = len(track_ids)
    labeled_count = len(label_mem)
    st.metric("Library Progress", f"{labeled_count}/{total_count}", delta=f"{labeled_count/total_count:.1%}")
    if ai_energy is not None:
        st.progress(min(max(ai_energy / 10.0, 0.0), 1.0))
        st.caption(f"Estimated energy: {ai_energy:.1f} / 10")
    if st.button("⏭️ Skip Track"):
        st.session_state.reload_track = True
        st.rerun()

# --- CENTER: Tagging Interface ---
with col_center:
    st.markdown("### 🏷️ Tagging")
    existing_tags = get_existing_tags()

    selected_existing = st.multiselect(
        "Genres",
        options=existing_tags,
        default=st.session_state.selected_tags
    )

    new_tags_input = st.text_input("New Genres (comma separated)", key="new_genre_input")

    st.markdown("### 🎭 Vibes")
    existing_vibes = get_existing_vibes()

    selected_vibes_ui = st.multiselect(
        "Vibes",
        options=existing_vibes,
        default=st.session_state.selected_vibes
    )

    new_vibes_input = st.text_input("New Vibes", key="new_vibe_input")

    st.markdown("---")
    energy = st.slider("⚡ Energy Level", 1, 10, 5)

    if st.button("✅ Save & Next", type="primary", use_container_width=True):
        final_tags = list(selected_existing)
        if new_tags_input: final_tags.extend([t.strip() for t in new_tags_input.split(",") if t.strip()])

        final_vibes = list(selected_vibes_ui)
        if new_vibes_input: final_vibes.extend([v.strip() for v in new_vibes_input.split(",") if v.strip()])

        if not final_tags:
            st.error("Add at least one genre!")
        else:
            save_to_db(current_track["trackid"], list(set(final_tags)), energy, list(set(final_vibes)))
            st.session_state.selected_tags = []
            st.session_state.selected_vibes = []
            st.session_state.reload_track = True
            st.toast("Saved!", icon="💾")
            st.rerun()

# --- RIGHT: AI Brain ---
with col_right:
    st.markdown("### 🧠 AI Brain")
    if not ai_genres and not ai_vibes:
        st.info("Not enough data for predictions yet.")

    if ai_genres:
        st.markdown("**Suggested Genres**")
        for tag, score in ai_genres[:5]:
            color = "green" if score > 0.7 else "orange" if score > 0.5 else "grey"
            c1, c2 = st.columns([3, 1])
            c1.markdown(f":{color}[{tag}] `{score:.0%}`")
            if tag not in selected_existing:
                if c2.button("➕", key=f"add_g_{tag}"):
                    st.session_state.selected_tags = selected_existing + [tag]
                    st.rerun()
            else:
                c2.caption("✅")

    if ai_vibes:
        st.markdown("---")
        st.markdown("**Suggested Vibes**")
        for tag, score in ai_vibes[:5]:
            color = "blue" if score > 0.7 else "violet"
            c1, c2 = st.columns([3, 1])
            c1.markdown(f":{color}[{tag}] `{score:.0%}`")
            if tag not in selected_vibes_ui:
                if c2.button("➕", key=f"add_v_{tag}"):
                    st.session_state.selected_vibes = selected_vibes_ui + [tag]
                    st.rerun()
            else:
                c2.caption("✅")