"""
Ingestor.py - Complete Track Ingestion Pipeline
Orchestrates TrackEmbedder and MetadataExtractor
Uploads to Supabase

ARCHITECTURE:
- TrackEmbedder.py: Computes 1280-dim embeddings (full track)
- MetadataExtractor.py: Extracts BPM and key (Essentia primary, librosa fallback)
- Ingestor.py (this file): Orchestrates both + reads tags + uploads
"""

import os
from pathlib import Path
from typing import List
import logging
from supabase import create_client, Client
from mutagen.mp4 import MP4
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3

# Import our custom modules
from TrackEmbedder import compute_embedding, validate_embedding
from MetadataExtractor import extract_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()  # loads variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# ------------------ Tag Reading (Title/Artist) ------------------

def read_tags(filepath: str):
    """
    Extract title and artist from audio file tags

    Args:
        filepath: Path to audio file

    Returns:
        Tuple of (title, artist)
    """
    ext = Path(filepath).suffix.lower()
    title, artist = None, None

    try:
        if ext == '.m4a':
            audiofile = MP4(filepath)
            title = audiofile.tags.get('\xa9nam', [None])[0]
            artist = audiofile.tags.get('\xa9ART', [None])[0]
        elif ext == '.mp3':
            audiofile = MP3(filepath, ID3=EasyID3)
            if audiofile.tags:
                title = audiofile.tags.get('title', [None])[0]
                artist = audiofile.tags.get('artist', [None])[0]
    except Exception as e:
        logger.warning(f"Failed to read tags from {filepath}: {e}")

    # Fallback to filename if no title
    if not title:
        title = Path(filepath).stem

    return title, artist


# ------------------ Supabase Operations ------------------

def track_exists_in_db(filepath: str) -> bool:
    """Check if track already exists in database"""
    try:
        result = supabase.table('tracks').select('trackid').eq('filepath', filepath).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking if track exists: {e}")
        return False


def upload_track_to_supabase(title: str, artist: str, filepath: str,
                             bpm: float, key: str, embedding: list) -> int:
    """
    Upload track to Supabase

    Args:
        title: Track title
        artist: Track artist
        filepath: Full path to audio file
        bpm: Beats per minute
        key: Musical key
        embedding: 1280-dim embedding as list

    Returns:
        trackid if successful, None otherwise
    """
    try:
        track_data = {
            'title': title,
            'artist': artist if artist else 'Unknown',
            'filepath': filepath,
            'embedding': embedding,
            'bpm': float(bpm) if bpm else None,
            'key': key
        }

        # Insert into tracks table
        result = supabase.table('tracks').insert(track_data).execute()

        if result.data:
            trackid = result.data[0]['trackid']
            logger.info(f"✓ Uploaded track: {title} (ID: {trackid})")

            # Initialize label entry
            label_data = {
                'trackid': trackid,
                'semantic_tags': [],
                'energy': None,
                'vibe': []
            }
            supabase.table('track_labels').insert(label_data).execute()

            return trackid
        else:
            logger.error(f"Upload failed for: {title}")
            return None

    except Exception as e:
        logger.error(f"Supabase upload error for {title}: {e}")
        return None


# ------------------ Main Pipeline ------------------

def get_audio_files(folder_path: str) -> List[str]:
    """Recursively find all audio files in folder"""
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    return audio_files


def ingest_single_track(filepath: str) -> bool:
    """
    Ingest a single track

    Args:
        filepath: Path to audio file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing: {Path(filepath).name}")

    # Step 1: Read tags (title, artist)
    title, artist = read_tags(filepath)
    logger.info(f"  Title: {title}")
    logger.info(f"  Artist: {artist}")

    # Step 2: Extract metadata (BPM, key) using MetadataExtractor
    metadata = extract_metadata(filepath)
    bpm = metadata.get('bpm')
    key = metadata.get('key')

    if not bpm:
        logger.warning("  BPM extraction failed")
    if not key:
        logger.warning("  Key extraction failed")

    logger.info(f"  BPM: {bpm}, Key: {key}")

    # Step 3: Compute embedding using TrackEmbedder
    embedding = compute_embedding(filepath, strategy='full')

    if embedding is None:
        logger.error("  ✗ Failed to compute embedding")
        return False

    # Step 4: Validate embedding
    if not validate_embedding(embedding):
        logger.error("  ✗ Invalid embedding")
        return False

    logger.info(f"  ✓ Valid embedding: shape {embedding.shape}")

    # Step 5: Upload to Supabase
    trackid = upload_track_to_supabase(
        title=title,
        artist=artist,
        filepath=filepath,
        bpm=bpm,
        key=key,
        embedding=embedding.tolist()
    )

    if trackid:
        return True
    else:
        logger.error("  ✗ Upload failed")
        return False


def ingest_tracks(folder_path: str, skip_existing: bool = True):
    """
    Complete ingestion pipeline for a folder

    Args:
        folder_path: Path to folder containing audio files
        skip_existing: If True, skip tracks already in database
    """
    logger.info("="*70)
    logger.info("TRACK INGESTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Input folder: {folder_path}")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info("")

    # Get all audio files
    audio_files = get_audio_files(folder_path)
    logger.info(f"Found {len(audio_files)} audio files\n")

    if not audio_files:
        logger.warning("No audio files found!")
        return

    stats = {
        'total': len(audio_files),
        'uploaded': 0,
        'skipped': 0,
        'failed': 0
    }

    # Process each file
    for i, filepath in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{len(audio_files)}] {'-'*50}")

        try:
            # Check if already exists
            if skip_existing and track_exists_in_db(filepath):
                logger.info(f"⊘ Track already exists in database, skipping")
                stats['skipped'] += 1
                continue

            # Ingest the track
            success = ingest_single_track(filepath)

            if success:
                stats['uploaded'] += 1
            else:
                stats['failed'] += 1

        except Exception as e:
            logger.error(f"✗ Unexpected error processing {filepath}: {e}")
            stats['failed'] += 1

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("INGESTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total files found: {stats['total']}")
    logger.info(f"✓ Successfully uploaded: {stats['uploaded']}")
    logger.info(f"⊘ Skipped (already in DB): {stats['skipped']}")
    logger.info(f"✗ Failed: {stats['failed']}")
    logger.info("="*70)


# ------------------ CLI Interface ------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Ingestor - Complete Track Ingestion Pipeline")
        print("\nUsage:")
        print("  python Ingestor.py <folder_path>")
        print("\nExample:")
        print("  python Ingestor.py /path/to/music")
        print("\nWhat it does:")
        print("  1. Reads title/artist from tags")
        print("  2. Extracts BPM and key (MetadataExtractor)")
        print("  3. Computes embeddings (TrackEmbedder)")
        print("  4. Uploads to Supabase")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    ingest_tracks(folder_path)