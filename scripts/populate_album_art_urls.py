"""
populate_album_art_urls.py
--------------------------
Finds every track with a null album_art_url and writes the expected
Supabase storage URL (artist_title.jpg) into the column.

No file reading or uploading — just URL construction + DB update.
If a file isn't actually in the bucket the frontend falls back to
iTunes → generative initials automatically.

Run: python populate_album_art_urls.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import make_cover_filename, get_supabase

load_dotenv()

supabase = get_supabase()

SUPABASE_URL = os.getenv("SUPABASE_URL")
COVERS_BASE = f"{SUPABASE_URL}/storage/v1/object/public/album-covers/"


def create_safe_filename(artist: str, title: str) -> str:
    """Build a safe ASCII storage filename. Delegates to shared_utils."""
    return make_cover_filename(artist, title, ext="")


def main():
    print("Fetching tracks with null album_art_url …")
    resp = supabase.table("tracks").select("trackid, artist, title").is_("album_art_url", "null").execute()
    tracks = resp.data
    print(f"  {len(tracks)} tracks to update\n")

    updated = 0
    skipped = 0

    for track in tracks:
        artist = (track.get("artist") or "unknown").strip()
        title  = (track.get("title")  or "").strip()

        if not title:
            skipped += 1
            continue

        filename = create_safe_filename(artist, title)
        url      = f"{COVERS_BASE}{filename}.jpg"

        try:
            supabase.table("tracks").update({"album_art_url": url}).eq("trackid", track["trackid"]).execute()
            updated += 1
            if updated % 50 == 0:
                print(f"  {updated} updated …")
        except Exception as e:
            print(f"  Error on trackid {track['trackid']} ({artist} — {title}): {e}")
            skipped += 1

    print(f"\nDone — updated: {updated}  skipped (no title): {skipped}")


if __name__ == "__main__":
    main()
