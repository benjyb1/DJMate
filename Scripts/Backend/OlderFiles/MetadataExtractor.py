"""
MetadataExtractor.py - Fixed Version
Extracts metadata (BPM, key, tags) using Librosa PRIMARY, Essentia fallback

FIXES APPLIED:
- Fixed librosa numpy float formatting error
- Proper type conversion for BPM values
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4

import essentia.standard as es
import librosa
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------ Metadata Extraction ------------------

def extract_metadata_from_tags(filepath: str) -> Dict:
    """Extract metadata from audio file tags"""
    try:
        suffix = Path(filepath).suffix.lower()
        if suffix == '.mp3':
            audio = MP3(filepath, ID3=EasyID3)
            return {
                'title': audio.get('title', ['Unknown'])[0],
                'artist': audio.get('artist', ['Unknown'])[0],
                'bpm': audio.get('bpm', [None])[0],
                'key': audio.get('initialkey', [None])[0]
            }
        elif suffix == '.m4a':
            audio = MP4(filepath)
            tags = audio.tags
            title = tags.get('\xa9nam', ['Unknown'])[0] if tags else 'Unknown'
            artist = tags.get('\xa9ART', ['Unknown'])[0] if tags else 'Unknown'
            bpm = tags.get('tmpo', [None])[0] if tags else None
            key = tags.get('key', [None])[0] if tags else None
            return {
                'title': title,
                'artist': artist,
                'bpm': bpm,
                'key': key
            }
        else:
            # For other formats, fallback to MP3 EasyID3 attempt
            audio = MP3(filepath, ID3=EasyID3)
            return {
                'title': audio.get('title', ['Unknown'])[0],
                'artist': audio.get('artist', ['Unknown'])[0],
                'bpm': audio.get('bpm', [None])[0],
                'key': audio.get('initialkey', [None])[0]
            }
    except Exception as e:
        logger.warning(f"Could not extract tags from {filepath}: {e}")
        return {'title': None, 'artist': None, 'bpm': None, 'key': None}


def analyze_audio(filepath: str) -> Dict:
    """
    Analyze audio to extract BPM and key

    BPM: Librosa PRIMARY, Essentia fallback (librosa is more robust)
    Key: Essentia only (best for musical key detection)
    """
    bpm = None
    key_str = None
    key_strength = None

    # ================== BPM DETECTION ==================
    # Strategy: Try librosa FIRST (primary), fall back to Essentia only if needed

    # Step 1: Try librosa (PRIMARY)
    try:
        logger.info(f"Attempting BPM detection with librosa...")
        y, sr = librosa.load(filepath, duration=60, mono=True)  # Load first 60s for speed
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        # CRITICAL FIX: Convert numpy scalar to Python float to avoid format errors
        tempo = float(tempo)

        # Validate librosa result
        if tempo and 40 <= tempo <= 250:
            bpm = round(tempo, 2)
            logger.info(f"✓ Librosa BPM: {bpm:.2f}")
        else:
            logger.warning(f"Librosa BPM outside valid range ({tempo:.2f} BPM), will try Essentia")

    except Exception as e:
        logger.warning(f"Librosa BPM detection failed: {e}, will try Essentia")

    # Step 2: Fallback to Essentia ONLY if librosa failed
    if bpm is None:
        try:
            logger.info(f"Falling back to Essentia for BPM...")
            audio = es.MonoLoader(filename=filepath, sampleRate=44100)()
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            essentia_bpm, _, _, _, _ = rhythm_extractor(audio)

            # CRITICAL FIX: Convert to Python float
            essentia_bpm = float(essentia_bpm)

            if essentia_bpm and 40 <= essentia_bpm <= 250:
                bpm = round(essentia_bpm, 2)
                logger.info(f"✓ Essentia BPM (fallback): {bpm:.2f}")
            else:
                logger.warning(f"Essentia BPM also outside valid range ({essentia_bpm:.2f})")

        except Exception as e:
            logger.error(f"Essentia BPM detection also failed: {e}")

    if bpm is None:
        logger.error(f"Could not detect BPM using either method")

    # ================== KEY DETECTION ==================
    # Key detection with Essentia (always use Essentia for this - best for musical key)
    try:
        audio_for_key = es.MonoLoader(filename=filepath, sampleRate=44100)()
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio_for_key)

        if key and scale:
            key_str = f"{key} {scale}"
            key_strength = float(strength)
            logger.info(f"✓ Key: {key_str} (strength: {strength:.2f})")
        else:
            logger.warning("Key extraction returned empty result")

    except Exception as e:
        logger.error(f"Key extraction failed: {e}")

    return {
        'bpm': bpm,
        'key': key_str,
        'key_strength': key_strength
    }


def extract_metadata(filepath: str) -> Dict:
    """
    Main entry point - extracts BPM and key from audio file
    This is what Ingestor.py expects to import

    Returns:
        dict with keys: bpm, key, key_strength
    """
    return analyze_audio(filepath)


def get_complete_metadata(filepath: str) -> Dict:
    """Get complete metadata using tags + audio analysis fallback"""
    metadata = {
        'filepath': filepath,
        'filename': Path(filepath).name,
        'file_hash': get_file_hash(filepath)
    }

    # 1. Try file tags first (fastest, most reliable)
    tags = extract_metadata_from_tags(filepath)
    metadata.update(tags)

    # 2. If BPM/key missing, analyze audio
    needs_analysis = not metadata.get('bpm') or not metadata.get('key')

    if needs_analysis:
        logger.info(f"Running audio analysis for {Path(filepath).name}")
        analysis = analyze_audio(filepath)
        metadata['bpm'] = metadata.get('bpm') or analysis['bpm']
        metadata['key'] = metadata.get('key') or analysis['key']
        metadata['key_strength'] = analysis.get('key_strength')

    # 3. Extract filename as fallback for title
    if not metadata.get('title') or metadata['title'] == 'Unknown':
        metadata['title'] = Path(filepath).stem

    return metadata


def get_file_hash(filepath: str) -> str:
    """Generate SHA256 hash of file for duplicate detection"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ------------------ File Discovery ------------------

def get_audio_files(folder_path: str) -> List[str]:
    """Recursively find all audio files in folder"""
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    return audio_files


# ------------------ Analysis Pipeline ------------------

def analyze_tracks(folder_path: str, output_file: str = "analysis_results.json"):
    """
    Analyze all tracks in folder and save results to JSON

    Args:
        folder_path: Path to folder containing audio files
        output_file: Path to output JSON file
    """
    logger.info(f"Starting audio analysis from: {folder_path}")

    # Get all audio files
    audio_files = get_audio_files(folder_path)
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.warning("No audio files found!")
        return

    results = []
    stats = {
        'total': len(audio_files),
        'analyzed': 0,
        'failed': 0
    }

    for i, filepath in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{len(audio_files)}] Analyzing: {Path(filepath).name}")

        try:
            # Extract complete metadata
            metadata = get_complete_metadata(filepath)

            logger.info(f"  Title: {metadata.get('title')}")
            logger.info(f"  Artist: {metadata.get('artist')}")
            logger.info(f"  BPM: {metadata.get('bpm')}, Key: {metadata.get('key')}")

            results.append(metadata)
            stats['analyzed'] += 1

        except Exception as e:
            logger.error(f"✗ Error analyzing {filepath}: {e}")
            stats['failed'] += 1

    # Save results to JSON
    logger.info(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump({
            'stats': stats,
            'tracks': results
        }, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"Analyzed: {stats['analyzed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Results saved to: {output_file}")

    return output_file


# ------------------ CLI Interface ------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python MetadataExtractor.py <folder_path> [output_file]")
        print("\nExample:")
        print("  python MetadataExtractor.py /path/to/music")
        print("  python MetadataExtractor.py /path/to/music my_analysis.json")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "analysis_results.json"

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    analyze_tracks(folder_path, output_file)