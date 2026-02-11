"""
Streamlit UI for Track Ingestion
Drag-and-drop interface for adding new tracks to Supabase
"""

import streamlit as st
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import logging
from supabase import create_client, Client
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
import essentia.standard as es
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Supabase setup ------------------
SUPABASE_URL = 'https://cvermotfxamubejfnoje.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag'


@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = get_supabase_client()

# ------------------ Load Model ------------------
MODEL_PATH = '/mnt/user-data/uploads/1769604462767_discogs-effnet-bs64.pb'


@st.cache_resource
def load_effnet_model():
    """Load the Discogs-EffNet model using Essentia's TensorflowPredictEffnetDiscogs"""
    try:
        model = es.TensorflowPredictEffnetDiscogs(modelFilename=MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


EFFNET_MODEL = load_effnet_model()


# ------------------ Helper Functions ------------------

def extract_metadata(filepath: str) -> Dict:
    """Extract metadata from audio file tags"""
    try:
        audio = MP3(filepath, ID3=EasyID3)
        return {
            'title': audio.get('title', ['Unknown'])[0],
            'artist': audio.get('artist', ['Unknown'])[0],
            'bpm': audio.get('bpm', [None])[0],
            'key': audio.get('initialkey', [None])[0]
        }
    except:
        return {'title': None, 'artist': None, 'bpm': None, 'key': None}


def analyze_audio(filepath: str) -> Dict:
    """Analyze audio to extract BPM and key using Essentia"""
    try:
        # Load audio (first 60 seconds)
        loader = es.MonoLoader(filename=filepath, startTime=0, duration=60)
        audio = loader()

        # BPM detection using Essentia RhythmExtractor2013
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio)

        # Key detection
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)

        return {
            'bpm': round(float(bpm), 2),
            'key': f"{key} {scale}"
        }
    except:
        return {'bpm': None, 'key': None}


def get_complete_metadata(filepath: str) -> Dict:
    """Get complete metadata using tags + audio analysis fallback"""
    metadata = {'filepath': filepath}

    # Try file tags first
    tags = extract_metadata(filepath)
    metadata.update(tags)

    # If BPM/key missing, analyze audio
    if not metadata.get('bpm') or not metadata.get('key'):
        analysis = analyze_audio(filepath)
        metadata['bpm'] = metadata.get('bpm') or analysis['bpm']
        metadata['key'] = metadata.get('key') or analysis['key']

    # Use filename as fallback title
    if not metadata.get('title') or metadata['title'] == 'Unknown':
        metadata['title'] = Path(filepath).stem

    return metadata


def compute_embedding(filepath: str) -> Optional[np.ndarray]:
    """Compute Discogs-EffNet embedding using Essentia's TensorflowPredictEffnetDiscogs"""
    try:
        loader = es.MonoLoader(filename=filepath, sampleRate=16000)
        audio = loader()
        embedding = EFFNET_MODEL(audio)
        return np.array(embedding).flatten()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def track_exists_in_db(filepath: str) -> bool:
    """Check if track already exists in database"""
    try:
        result = supabase.table('tracks').select('trackid').eq('filepath', filepath).execute()
        return len(result.data) > 0
    except:
        return False


def upload_track_to_supabase(metadata: Dict, embedding: np.ndarray) -> Optional[str]:
    """Upload track to Supabase"""
    try:
        track_data = {
            'title': metadata.get('title', 'Unknown'),
            'artist': metadata.get('artist', 'Unknown'),
            'filepath': metadata['filepath'],
            'embedding': embedding.tolist(),
            'bpm': float(metadata['bpm']) if metadata.get('bpm') else None,
            'key': metadata.get('key')
        }

        result = supabase.table('tracks').insert(track_data).execute()

        if result.data:
            trackid = result.data[0]['trackid']

            # Initialize label entry
            label_data = {
                'trackid': trackid,
                'semantic_tags': [],
                'energy': None,
                'vibe': []
            }
            supabase.table('track_labels').insert(label_data).execute()

            return trackid
        return None
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return None


def get_audio_files(folder_path: str) -> List[str]:
    """Recursively find all audio files"""
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    return audio_files


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Track Ingestion", page_icon="🎵", layout="wide")

st.title("🎵 Track Ingestion Pipeline")
st.markdown("Add new tracks to your AI DJ database")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    skip_existing = st.checkbox("Skip existing tracks", value=True)
    show_details = st.checkbox("Show detailed logs", value=True)

    st.divider()
    st.markdown("### 📊 Database Stats")
    try:
        track_count = supabase.table('tracks').select('trackid', count='exact').execute()
        st.metric("Total Tracks", track_count.count if hasattr(track_count, 'count') else "N/A")
    except:
        st.error("Could not fetch stats")

# Main interface
folder_path = st.text_input(
    "📁 Folder Path",
    placeholder="/path/to/your/music/folder",
    help="Enter the full path to the folder containing your audio files"
)

col1, col2 = st.columns([1, 3])

with col1:
    start_button = st.button("🚀 Start Ingestion", type="primary", use_container_width=True)

with col2:
    if os.path.exists(folder_path) if folder_path else False:
        audio_files = get_audio_files(folder_path)
        st.info(f"Found {len(audio_files)} audio files in folder")
    elif folder_path:
        st.error("Folder not found")

# Processing
if start_button and folder_path:
    if not os.path.exists(folder_path):
        st.error("❌ Folder does not exist")
    elif EFFNET_MODEL is None:
        st.error("❌ Model not loaded")
    else:
        audio_files = get_audio_files(folder_path)

        if not audio_files:
            st.warning("No audio files found in folder")
        else:
            st.success(f"Starting ingestion of {len(audio_files)} files...")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Stats
            stats = {'uploaded': 0, 'skipped': 0, 'failed': 0}
            stats_container = st.container()

            # Log container
            if show_details:
                log_container = st.expander("📋 Detailed Logs", expanded=True)

            # Process each file
            for i, filepath in enumerate(audio_files):
                progress = (i + 1) / len(audio_files)
                progress_bar.progress(progress)

                filename = Path(filepath).name
                status_text.text(f"Processing {i + 1}/{len(audio_files)}: {filename}")

                try:
                    # Check if exists
                    if skip_existing and track_exists_in_db(filepath):
                        stats['skipped'] += 1
                        if show_details:
                            with log_container:
                                st.info(f"⊘ Skipped (exists): {filename}")
                        continue

                    # Extract metadata
                    metadata = get_complete_metadata(filepath)

                    # Compute embedding
                    embedding = compute_embedding(filepath)
                    if embedding is None:
                        stats['failed'] += 1
                        if show_details:
                            with log_container:
                                st.error(f"✗ Failed (embedding): {filename}")
                        continue

                    # Upload
                    trackid = upload_track_to_supabase(metadata, embedding)
                    if trackid:
                        stats['uploaded'] += 1
                        if show_details:
                            with log_container:
                                st.success(f"✓ Uploaded: {metadata['title']} by {metadata['artist']}")
                    else:
                        stats['failed'] += 1
                        if show_details:
                            with log_container:
                                st.error(f"✗ Failed (upload): {filename}")

                except Exception as e:
                    stats['failed'] += 1
                    if show_details:
                        with log_container:
                            st.error(f"✗ Error: {filename} - {str(e)}")

                # Update stats display
                with stats_container:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("✓ Uploaded", stats['uploaded'])
                    col2.metric("⊘ Skipped", stats['skipped'])
                    col3.metric("✗ Failed", stats['failed'])

            # Final summary
            progress_bar.progress(1.0)
            status_text.text("✅ Ingestion complete!")

            st.balloons()

            st.success(f"""
            ### 🎉 Ingestion Complete!
            - **Uploaded:** {stats['uploaded']} tracks
            - **Skipped:** {stats['skipped']} tracks (already in database)
            - **Failed:** {stats['failed']} tracks
            """)

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Enter the folder path** containing your audio files
    2. **Configure settings** in the sidebar (optional)
    3. **Click "Start Ingestion"** to begin processing

    The pipeline will:
    - Recursively scan for audio files (mp3, wav, flac, m4a, aac, ogg)
    - Extract metadata (title, artist, BPM, key)
    - Compute audio embeddings using Discogs-EffNet
    - Upload to Supabase (skipping duplicates)
    - Initialize empty label entries for future tagging

    **Note:** Processing 500 tracks may take 30-60 minutes depending on your hardware.
    """)