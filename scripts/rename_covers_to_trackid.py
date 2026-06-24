"""
Rename all album covers in Supabase storage from artist_title.jpg → {trackid}.jpg

This eliminates all filename generation logic — trackid is unique and has no
special characters. After running, album_art_url in the DB is updated to match.
"""
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = "album-covers"
BASE = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    sys.exit(1)

from supabase import create_client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}


def list_all_bucket_files():
    """Get all filenames currently in the bucket."""
    all_files = set()
    offset = 0
    while True:
        files = sb.storage.from_(BUCKET).list("", {"limit": 1000, "offset": offset})
        if not files:
            break
        for f in files:
            name = f.get("name", "")
            if name and name != ".emptyFolderPlaceholder":
                all_files.add(name)
        if len(files) < 1000:
            break
        offset += len(files)
    return all_files


def main():
    # 1. Get all tracks with album_art_url
    resp = sb.table("tracks").select("trackid, title, artist, album_art_url").execute()
    tracks = resp.data or []
    print(f"{len(tracks)} tracks in DB")

    # 2. Get existing bucket files for quick lookup
    print("Listing bucket files...")
    bucket_files = list_all_bucket_files()
    print(f"{len(bucket_files)} files in bucket")

    renamed = 0
    skipped = 0
    missing = 0
    errors = 0

    for i, track in enumerate(tracks):
        tid = str(track["trackid"])
        new_filename = f"{tid}.jpg"
        new_url = f"{BASE}{new_filename}"

        # Already renamed?
        if track.get("album_art_url") == new_url:
            skipped += 1
            continue

        # Already exists in bucket with new name?
        if new_filename in bucket_files:
            # Just update the DB URL
            sb.table("tracks").update({"album_art_url": new_url}).eq("trackid", tid).execute()
            renamed += 1
            if (renamed % 50) == 0:
                print(f"  Progress: {renamed} renamed, {i+1}/{len(tracks)} processed")
            continue

        # Find current file in bucket
        old_url = track.get("album_art_url") or ""
        if not old_url:
            missing += 1
            continue

        old_filename = old_url.split("/")[-1] if "/" in old_url else ""
        if not old_filename or old_filename not in bucket_files:
            missing += 1
            continue

        # Download old file, upload with new name
        try:
            img_bytes = sb.storage.from_(BUCKET).download(old_filename)
            if not img_bytes:
                missing += 1
                continue

            # Upload with new name
            sb.storage.from_(BUCKET).upload(
                new_filename,
                img_bytes,
                {"content-type": "image/jpeg", "upsert": "true"},
            )

            # Update DB
            sb.table("tracks").update({"album_art_url": new_url}).eq("trackid", tid).execute()

            # Delete old file
            sb.storage.from_(BUCKET).remove([old_filename])
            bucket_files.discard(old_filename)
            bucket_files.add(new_filename)

            renamed += 1
            if (renamed % 50) == 0:
                print(f"  Progress: {renamed} renamed, {i+1}/{len(tracks)} processed")

        except Exception as e:
            print(f"  Error on {tid} ({track.get('artist')} - {track.get('title')}): {e}")
            errors += 1

    print(f"\nDone. Renamed: {renamed}  Skipped (already done): {skipped}  "
          f"Missing source: {missing}  Errors: {errors}")


if __name__ == "__main__":
    main()
