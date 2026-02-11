import os
import json
import time
from pathlib import Path
from supabase import create_client, Client
from mutagen import File
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

# ==================== CONFIGURATION ====================
SUPABASE_URL = "https://cvermotfxamubejfnoje.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"
BUCKET_NAME = "album-covers"
MUSIC_FOLDER = Path.home() / "Desktop" / "Mixing"
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.mp4', '.ogg', '.wma'}
PROGRESS_FILE = "upload_progress_tracks.json"

# ==================== SETUP ====================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== UTILITIES ====================

def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def normalise_artist(artist):
    if not artist or not isinstance(artist, str) or not artist.strip():
        return "unknown artist"
    return artist.strip()

def normalise_title(title):
    if not title or not isinstance(title, str) or not title.strip():
        return None
    return title.strip()

def create_safe_filename(artist, title):
    safe_name = f"{artist}_{title}".lower()
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in safe_name)
    safe_name = '_'.join(filter(None, safe_name.split('_')))
    return safe_name[:150]

# ==================== ALBUM ART EXTRACTION ====================

def extract_album_art(audio_path):
    try:
        audio = File(audio_path)
        if audio is None:
            return None

        if isinstance(audio, MP3):
            for tag in audio.tags.values():
                if hasattr(tag, 'mime') and 'image' in tag.mime:
                    return tag.data

        elif isinstance(audio, FLAC):
            if audio.pictures:
                return audio.pictures[0].data

        elif isinstance(audio, MP4):
            if 'covr' in audio.tags:
                return bytes(audio.tags['covr'][0])

    except Exception:
        pass

    return None

def get_track_info(audio_path):
    try:
        audio = File(audio_path)
        if audio is None or not hasattr(audio, 'tags') or not audio.tags:
            return None, None

        artist = None
        title = None

        artist_keys = ['artist', 'ARTIST', 'TPE1', '©ART', 'Artist', 'TPE2', 'albumartist']
        title_keys = ['title', 'TITLE', 'TIT2', '©nam', 'Title']

        for key in artist_keys:
            if key in audio.tags:
                val = audio.tags[key]
                artist = str(val[0]) if isinstance(val, list) else str(val)
                break

        for key in title_keys:
            if key in audio.tags:
                val = audio.tags[key]
                title = str(val[0]) if isinstance(val, list) else str(val)
                break

        return normalise_artist(artist), normalise_title(title)

    except Exception:
        return None, None

# ==================== UPLOAD ====================

def upload_album_art(image_data, filename):
    try:
        file_path = f"{filename}.jpg"

        supabase.storage.from_(BUCKET_NAME).upload(
            file_path,
            image_data,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )

        url_response = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        return url_response['data']['publicUrl']

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

# ==================== MAIN PROCESS ====================

def process_all_tracks():
    print(f"Scanning: {MUSIC_FOLDER}\n")

    progress = load_progress()
    track_art_map = progress.get('uploaded_tracks', {})

    # Step 1: Ensure missing-artist keys are normalised
    normalised_map = {}
    for key, value in track_art_map.items():
        parts = key.split("_", 1)
        if len(parts) == 2:
            artist, title = parts
        else:
            continue

        artist = normalise_artist(artist)
        title = normalise_title(title)

        if title:
            new_key = f"{artist}_{title}".lower()
            normalised_map[new_key] = value

    track_art_map = normalised_map
    save_progress({'uploaded_tracks': track_art_map})

    print(f"Loaded {len(track_art_map)} tracks from progress file")

    # Step 2: Update database
    response = supabase.table('tracks').select('*').execute()
    db_tracks = response.data

    print(f"Found {len(db_tracks)} tracks in database")

    updated_count = 0
    not_found = 0

    for track in db_tracks:
        artist = normalise_artist(track.get('artist'))
        title = normalise_title(track.get('title'))

        if not title:
            not_found += 1
            continue

        track_key = f"{artist}_{title}".lower()

        if track_key in track_art_map:
            url = track_art_map[track_key]['url']

            try:
                supabase.table('tracks').update({
                    'album_art_url': url
                }).eq('trackid', track['trackid']).execute()

                updated_count += 1

                if updated_count % 50 == 0:
                    print(f"Updated {updated_count} tracks...")

            except Exception as e:
                print(f"❌ Error updating {track['trackid']}: {e}")
        else:
            not_found += 1

    print("\nUpdate complete")
    print(f"Updated: {updated_count}")
    print(f"No cover found for: {not_found}")

if __name__ == "__main__":
    process_all_tracks()