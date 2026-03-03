"""
fix_artist_titles.py
────────────────────
Scans tracks whose artist field is blank / unknown and whose title looks like
"Artist - Title".  Splits them out and writes the corrected artist + title
back to the DB.

Usage:
  python backend/fix_artist_titles.py            # dry-run (preview only)
  python backend/fix_artist_titles.py --write    # actually update the DB
"""

import os
import sys
import re
import argparse
from dotenv import load_dotenv

load_dotenv()

# Artist field values we treat as "unknown" (case-insensitive)
UNKNOWN_ARTIST_VALUES = {
    "", "unknown", "unknown artist", "various", "various artists", "n/a", "none",
}

# Regex: one or more chars, then " - ", then one or more chars
ARTIST_TITLE_PATTERN = re.compile(r"^(.+?)\s+-\s+(.+)$")


def get_supabase():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌  SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


def is_unknown_artist(artist: str | None) -> bool:
    if artist is None:
        return True
    return artist.strip().lower() in UNKNOWN_ARTIST_VALUES


def main():
    parser = argparse.ArgumentParser(
        description="Fix tracks where the title contains 'Artist - Title' and the artist is unknown."
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write corrections to the DB. Without this flag the script is a dry-run."
    )
    args = parser.parse_args()

    dry_run = not args.write
    if dry_run:
        print("🔍  DRY RUN — no changes will be written.  Pass --write to apply.\n")

    supabase = get_supabase()

    # ── Load all tracks ────────────────────────────────────────────────────────
    print("Loading tracks …")
    resp = supabase.table("tracks").select("trackid, title, artist, filepath").execute()
    tracks = resp.data or []
    print(f"  {len(tracks)} tracks total\n")

    # ── Find candidates ────────────────────────────────────────────────────────
    candidates = []
    for t in tracks:
        if not is_unknown_artist(t.get("artist")):
            continue                        # artist is already known, skip
        title = (t.get("title") or "").strip()
        if not title:
            continue
        m = ARTIST_TITLE_PATTERN.match(title)
        if not m:
            continue                        # title doesn't look like "Artist - Title"
        new_artist = m.group(1).strip()
        new_title  = m.group(2).strip()
        candidates.append({
            "trackid":    t["trackid"],
            "old_artist": t.get("artist"),
            "old_title":  title,
            "new_artist": new_artist,
            "new_title":  new_title,
        })

    if not candidates:
        print("✅  No tracks need fixing.")
        return

    print(f"Found {len(candidates)} track(s) to fix:\n")
    for c in candidates:
        print(f"  [trackid {c['trackid']}]")
        print(f"    artist : {repr(c['old_artist'])}  →  {repr(c['new_artist'])}")
        print(f"    title  : {repr(c['old_title'])}  →  {repr(c['new_title'])}")
        print()

    # ── Write corrections ──────────────────────────────────────────────────────
    if dry_run:
        print("─── Dry run complete — run with --write to apply these changes. ───")
        return

    print("Writing corrections …")
    updated = 0
    failed  = 0
    for c in candidates:
        try:
            supabase.table("tracks").update({
                "artist": c["new_artist"],
                "title":  c["new_title"],
            }).eq("trackid", c["trackid"]).execute()
            updated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to update {c['trackid']}: {e}")
            failed += 1

    print(f"\n✅  Done.  Updated: {updated}  Failed: {failed}")


if __name__ == "__main__":
    main()
