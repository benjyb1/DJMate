import os
import json
import unicodedata
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "audio-files"
PROGRESS_FILE = "upload_progress_audio.json"

MIME_MAP = {
    '.mp3':  'audio/mpeg',
    '.flac': 'audio/flac',
    '.m4a':  'audio/mp4',
    '.mp4':  'audio/mp4',
    '.ogg':  'audio/ogg',
    '.wav':  'audio/wav',
    '.aiff': 'audio/aiff',
    '.aif':  'audio/aiff',
}

# ==================== SETUP ===================
# Before running this script:
# 1. Create an "audio-files" bucket in Supabase Storage (set to public)
# 2. Add an "audio_url" column to the tracks table:
#    ALTER TABLE tracks ADD COLUMN audio_url TEXT;
# 3. Run: pip install supabase python-dotenv

# ==================== UTILITIES ====================

def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

# ==================== UPLOAD ====================

def upload_audio(audio_path: Path, trackid: str) -> str | None:
    ext = audio_path.suffix.lower()
    mime = MIME_MAP.get(ext, 'audio/mpeg')
    bucket_path = f"{trackid}{ext}"

    try:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        supabase.storage.from_(BUCKET_NAME).upload(
            bucket_path,
            audio_bytes,
            file_options={"content-type": mime, "upsert": "true"}
        )

        url_response = supabase.storage.from_(BUCKET_NAME).get_public_url(bucket_path)
        # Handle both dict and string responses depending on supabase-py version
        if isinstance(url_response, dict):
            return url_response.get('data', {}).get('publicUrl') or url_response.get('publicUrl')
        return url_response

    except Exception as e:
        print(f"  Upload error for trackid {trackid}: {e}")
        return None

# ==================== MAIN PROCESS ====================

def process_all_tracks():
    progress = load_progress()
    uploaded = set(progress.get('uploaded_trackids', []))
    print(f"Progress: {len(uploaded)} tracks already uploaded\n")

    # Fetch all tracks that have a filepath but no audio_url yet
    response = supabase.table('tracks').select('trackid, artist, title, filepath, audio_url').execute()
    db_tracks = response.data
    print(f"{len(db_tracks)} tracks in DB")

    to_upload = [
        t for t in db_tracks
        if t.get('filepath')
        and not t.get('audio_url')
        and str(t['trackid']) not in uploaded
    ]
    already_have_url = sum(1 for t in db_tracks if t.get('audio_url'))
    print(f"  {already_have_url} already have audio_url")
    print(f"  {len(to_upload)} need uploading\n")

    newly_uploaded = 0
    failed = 0

    for track in to_upload:
        trackid  = str(track['trackid'])
        filepath = track['filepath']
        artist   = track.get('artist', 'Unknown')
        title    = track.get('title', 'Unknown')

        audio_path = Path(filepath)
        if not audio_path.exists():
            print(f"  File not found: {filepath}")
            failed += 1
            continue

        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"  [{size_mb:5.1f}MB] {artist} — {title}")

        url = upload_audio(audio_path, trackid)
        if url:
            # Update DB immediately so we can resume safely
            try:
                supabase.table('tracks').update({'audio_url': url}).eq('trackid', track['trackid']).execute()
                uploaded.add(trackid)
                newly_uploaded += 1
                print(f"    ✓ Uploaded & DB updated")
            except Exception as e:
                print(f"    DB update error: {e}")
                failed += 1

            if newly_uploaded % 10 == 0:
                save_progress({'uploaded_trackids': list(uploaded)})
                print(f"  Progress saved ({newly_uploaded} uploaded so far)\n")
        else:
            failed += 1

    save_progress({'uploaded_trackids': list(uploaded)})
    print(f"\nDone.  Newly uploaded: {newly_uploaded}  Failed/missing: {failed}")


if __name__ == "__main__":
    process_all_tracks()
