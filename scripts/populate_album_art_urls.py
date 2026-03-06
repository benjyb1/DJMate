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
import unicodedata
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

COVERS_BASE = f"{SUPABASE_URL}/storage/v1/object/public/album-covers/"


def create_safe_filename(artist: str, title: str) -> str:
    """
    Mirrors create_safe_filename in upload_album_covers.py.
    Accented chars are stripped to their ASCII base (é → e, ü → u).
    Everything non-alphanumeric (except - _) becomes _.
    Consecutive underscores are collapsed. Max 150 chars.
    """
    raw = f"{artist}_{title}"
    # Decompose accents then drop the combining characters
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = raw.lower()
    raw = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    raw = "_".join(filter(None, raw.split("_")))
    return raw[:150]


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
