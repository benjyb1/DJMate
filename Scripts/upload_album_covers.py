import os
import json
import time
from pathlib import Path
from supabase import create_client, Client
from mutagen import File
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()  # loads variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "album-covers"
MUSIC_FOLDER = Path.home() / "Desktop" / "Mixing"
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.mp4', '.ogg', '.wma'}
PROGRESS_FILE = "upload_progress_tracks.json"

# ==================== SETUP ===================

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
    progress = load_progress()
    track_art_map = progress.get('uploaded_tracks', {})

    # Step 1: Normalise any legacy keys in the progress file
    normalised_map = {}
    for key, value in track_art_map.items():
        parts = key.split("_", 1)
        if len(parts) == 2:
            artist, title = parts
        else:
            continue
        artist = normalise_artist(artist)
        title  = normalise_title(title)
        if title:
            new_key = f"{artist}_{title}".lower()
            normalised_map[new_key] = value
    track_art_map = normalised_map
    save_progress({'uploaded_tracks': track_art_map})
    print(f"Progress file: {len(track_art_map)} covers already uploaded\n")

    # Step 2: Scan music folder and upload any missing covers
    print(f"Scanning: {MUSIC_FOLDER}")
    audio_files = [
        p for p in MUSIC_FOLDER.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    print(f"Found {len(audio_files)} audio files\n")

    newly_uploaded = 0
    scan_skipped   = 0
    scan_no_art    = 0

    for audio_path in audio_files:
        artist, title = get_track_info(audio_path)
        if not title:
            scan_no_art += 1
            continue

        track_key = f"{normalise_artist(artist)}_{title}".lower()

        if track_key in track_art_map:
            scan_skipped += 1
            continue  # already uploaded

        image_data = extract_album_art(audio_path)
        if not image_data:
            scan_no_art += 1
            continue

        filename = create_safe_filename(normalise_artist(artist), title)
        url = upload_album_art(image_data, filename)
        if url:
            track_art_map[track_key] = {'url': url}
            newly_uploaded += 1
            if newly_uploaded % 25 == 0:
                save_progress({'uploaded_tracks': track_art_map})
                print(f"  Uploaded {newly_uploaded} covers so far …")

    save_progress({'uploaded_tracks': track_art_map})
    print(f"\nScan complete — uploaded: {newly_uploaded}  "
          f"already done: {scan_skipped}  no art in file: {scan_no_art}\n")

    # Step 3: Update album_art_url column in DB for every track
    print("Updating database …")
    response  = supabase.table('tracks').select('trackid, artist, title').execute()
    db_tracks = response.data
    print(f"  {len(db_tracks)} tracks in DB")

    updated_count = 0
    not_found     = 0

    for track in db_tracks:
        artist = normalise_artist(track.get('artist'))
        title  = normalise_title(track.get('title'))
        if not title:
            not_found += 1
            continue

        track_key = f"{artist}_{title}".lower()
        if track_key not in track_art_map:
            not_found += 1
            continue

        url = track_art_map[track_key]['url']
        try:
            supabase.table('tracks').update({
                'album_art_url': url
            }).eq('trackid', track['trackid']).execute()
            updated_count += 1
            if updated_count % 50 == 0:
                print(f"  Updated {updated_count} tracks …")
        except Exception as e:
            print(f"  ❌ Error updating trackid {track['trackid']}: {e}")

    print(f"\n✅  Done.  DB updated: {updated_count}  No cover found: {not_found}")

if __name__ == "__main__":
    process_all_tracks()