import os
import streamlit as st
from supabase import create_client

from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()  # loads variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# --- 2. CONFIG ---
MUSIC_LIBRARY_ROOT = "/Users/benjyb/Desktop/Mixing"


# --- 3. HELPER FUNCTIONS ---

def get_stats():
    total_resp = supabase.table("tracks").select("trackid", count="exact").execute()
    tagged_resp = supabase.table("track_labels").select("trackid", count="exact").execute()
    return (total_resp.count or 0), (tagged_resp.count or 0)


def get_unique_tags(column_name):
    try:
        resp = supabase.table("track_labels").select(column_name).execute()
        unique_pool = set()
        for row in resp.data:
            tags = row.get(column_name)
            if tags and isinstance(tags, list):
                for t in tags: unique_pool.add(t)
        return sorted(list(unique_pool))
    except:
        return []


def get_unlabeled_track():
    labeled_resp = supabase.table("track_labels").select("trackid").execute()
    labeled_ids = [x["trackid"] for x in labeled_resp.data] if labeled_resp.data else []
    query = supabase.table("tracks").select("trackid, title, artist, filepath, bpm, key")
    if labeled_ids:
        id_list_string = f"({','.join(map(str, labeled_ids))})"
        query = query.not_.filter("trackid", "in", id_list_string)
    resp = query.limit(1).execute()
    return resp.data[0] if resp.data else None


def find_file_recursively(root_dir, target_filename):
    for root, dirs, files in os.walk(root_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


def save_to_db(trackid, genres, energy, vibes):
    label_data = {"trackid": trackid, "semantic_tags": genres, "energy": energy, "vibe": vibes}
    supabase.table("track_labels").upsert(label_data).execute()


def add_new_tag(tag_type, track_id):
    input_key = f"new_{tag_type}_input"
    selection_key = f"selected_{tag_type}_{track_id}"
    new_tag = st.session_state[input_key].strip()
    if new_tag:
        current_tags = list(st.session_state.get(selection_key, []))
        if new_tag not in current_tags:
            current_tags.append(new_tag)
            st.session_state[selection_key] = current_tags
    st.session_state[input_key] = ""


# --- 4. SESSION STATE INITIALIZATION ---
if "current_track" not in st.session_state or st.session_state.current_track is None:
    st.session_state.current_track = get_unlabeled_track()

# --- 5. UI CONFIG ---
st.set_page_config(layout="wide", page_title="AI DJ Trainer")

if st.session_state.current_track:
    track = st.session_state.current_track
    track_id = track['trackid']

    # Init state keys
    if f"selected_genre_{track_id}" not in st.session_state:
        st.session_state[f"selected_genre_{track_id}"] = []
    if f"selected_vibe_{track_id}" not in st.session_state:
        st.session_state[f"selected_vibe_{track_id}"] = []

    # --- SIDEBAR (AUDIO PLAYER & PROGRESS) ---
    with st.sidebar:
        st.title("🕹️ Controls")

        # Audio Player
        target_name = os.path.basename(track["filepath"])
        full_local_path = find_file_recursively(MUSIC_LIBRARY_ROOT, target_name)
        if full_local_path and os.path.exists(full_local_path):
            st.write("### Listen")
            with open(full_local_path, 'rb') as f:
                st.audio(f.read(), format='audio/mp3')
        else:
            st.error("Audio file not found locally.")

        st.divider()

        # Stats & Progress
        total_tracks, tagged_count = get_stats()
        st.write("### 📊 Library Stats")
        st.metric("Tagged", f"{tagged_count} / {total_tracks}")
        if total_tracks > 0:
            st.progress(tagged_count / total_tracks)

        st.divider()
        if st.button("⏭️ Skip Track", use_container_width=True):
            st.session_state.current_track = None
            st.rerun()

    # --- MAIN CONTENT (TRACK DATA & CLASSIFIER) ---
    # Track Title and Artist are now the Hero Section
    st.title(f"🎵 {track['title']}")
    st.subheader(f"by {track['artist']}")

    # Metadata Row
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("BPM", track['bpm'])
    m_col2.metric("Key", track['key'])
    m_col3.metric("Track ID", track_id)

    st.divider()

    # Classification Section
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🏷️ Genres")
        existing_genres = get_unique_tags("semantic_tags")
        genre_options = sorted(list(set(existing_genres + st.session_state[f"selected_genre_{track_id}"])))
        st.multiselect("Pick Genres:", options=genre_options, key=f"selected_genre_{track_id}",
                       label_visibility="collapsed")
        st.text_input("Add new genre...", key="new_genre_input", on_change=add_new_tag, args=("genre", track_id))

    with c2:
        st.markdown("### ✨ Vibes")
        existing_vibes = get_unique_tags("vibe")
        vibe_options = sorted(list(set(existing_vibes + st.session_state[f"selected_vibe_{track_id}"])))
        st.multiselect("Pick Vibes:", options=vibe_options, key=f"selected_vibe_{track_id}",
                       label_visibility="collapsed")
        st.text_input("Add new vibe...", key="new_vibe_input", on_change=add_new_tag, args=("vibe", track_id))

    st.write("")
    energy_level = st.select_slider("⚡ **Energy Level**", options=range(1, 11), value=5, key=f"energy_{track_id}")

    st.write("")
    if st.button("✅ Save & Load Next", use_container_width=True, type="primary"):
        final_genres = st.session_state[f"selected_genre_{track_id}"]
        final_vibes = st.session_state[f"selected_vibe_{track_id}"]
        if not final_genres:
            st.error("Please add at least one genre!")
        else:
            save_to_db(track_id, final_genres, energy_level, final_vibes)
            del st.session_state[f"selected_genre_{track_id}"]
            del st.session_state[f"selected_vibe_{track_id}"]
            st.session_state.current_track = None
            st.rerun()

else:
    st.balloons()
    st.success("🎉 All tracks labeled!")