import os
import json
import sys
from pathlib import Path
from mutagen import File
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import make_cover_filename, get_supabase

supabase = get_supabase()
BUCKET_NAME = "album-covers"
PROGRESS_FILE = "upload_progress_tracks.json"

# ==================== UTILITIES ====================

def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def create_safe_filename(artist, title):
    """Build a safe ASCII storage filename. Delegates to shared_utils."""
    return make_cover_filename(artist, title, ext="")

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
        if isinstance(url_response, dict):
            return url_response.get('data', {}).get('publicUrl') or url_response.get('publicUrl')
        return url_response

    except Exception as e:
        print(f"  Upload error: {e}")
        return None

# ==================== MAIN PROCESS ====================

def process_all_tracks():
    progress = load_progress()
    done_ids = set(progress.get('done_trackids', []))
    print(f"Progress: {len(done_ids)} tracks already processed\n")

    # Fetch all tracks that have a filepath but no album_art_url
    response = supabase.table('tracks').select('trackid, artist, title, filepath, album_art_url').execute()
    db_tracks = response.data
    print(f"{len(db_tracks)} tracks in DB")

    to_process = [
        t for t in db_tracks
        if t.get('filepath')
        and not t.get('album_art_url')
        and str(t['trackid']) not in done_ids
    ]
    already_have = sum(1 for t in db_tracks if t.get('album_art_url'))
    print(f"  {already_have} already have album_art_url")
    print(f"  {len(to_process)} need processing\n")

    uploaded   = 0
    no_art     = 0
    not_found  = 0

    for track in to_process:
        trackid  = str(track['trackid'])
        filepath = track['filepath']
        artist   = track.get('artist') or 'unknown_artist'
        title    = track.get('title')

        if not title:
            no_art += 1
            continue

        audio_path = Path(filepath)
        if not audio_path.exists():
            print(f"  File not found: {filepath}")
            not_found += 1
            continue

        image_data = extract_album_art(audio_path)
        if not image_data:
            # No embedded art in this file — mark as done so we skip it next run
            done_ids.add(trackid)
            no_art += 1
            continue

        filename = create_safe_filename(artist, title)
        url = upload_album_art(image_data, filename)

        if url:
            try:
                supabase.table('tracks').update({'album_art_url': url}).eq('trackid', track['trackid']).execute()
                done_ids.add(trackid)
                uploaded += 1
                print(f"  ✓ {artist} — {title}")
            except Exception as e:
                print(f"  DB error for {artist} — {title}: {e}")
        else:
            no_art += 1

        if uploaded % 25 == 0 and uploaded > 0:
            save_progress({'done_trackids': list(done_ids)})
            print(f"  Progress saved ({uploaded} uploaded so far)\n")

    save_progress({'done_trackids': list(done_ids)})
    print(f"\nDone.  Uploaded: {uploaded}  No embedded art: {no_art}  File not found: {not_found}")


if __name__ == "__main__":
    process_all_tracks()
