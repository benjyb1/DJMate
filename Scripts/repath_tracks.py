"""
repath_tracks.py — Re-scan a directory and update filepath in Supabase for every track.

The script:
  1. Fetches all tracks from Supabase (trackid, title, artist, filepath).
  2. Walks SEARCH_ROOT recursively and indexes every audio file by filename.
  3. For each track, tries to find its file:
       a. Exact filename match against the old filepath's basename.
       b. Fallback: case-insensitive match.
  4. Updates the DB with the new absolute path.
  5. Prints a summary of matched / unmatched tracks.

Usage:
    python Scripts/repath_tracks.py --root "/path/to/your/music"
    python Scripts/repath_tracks.py --root "/path/to/your/music" --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".aac"}


def build_file_index(root: str) -> dict[str, list[str]]:
    """Walk root and return {lowercase_filename: [absolute_path, ...]}."""
    index: dict[str, list[str]] = defaultdict(list)
    root_path = Path(root)
    if not root_path.exists():
        print(f"ERROR: Search root does not exist: {root}")
        sys.exit(1)

    print(f"Scanning {root} for audio files...")
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                abs_path = str(Path(dirpath) / fname)
                index[fname.lower()].append(abs_path)
                count += 1

    print(f"Found {count} audio files across {len(index)} unique filenames.\n")
    return dict(index)


def fetch_all_tracks(supabase) -> list[dict]:
    """Page through the tracks table and return all rows."""
    all_tracks = []
    page_size = 500
    offset = 0
    while True:
        resp = supabase.table("tracks") \
            .select("trackid, title, artist, filepath") \
            .range(offset, offset + page_size - 1) \
            .execute()
        batch = resp.data or []
        all_tracks.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return all_tracks


def resolve_new_path(track: dict, file_index: dict[str, list[str]]) -> str | None:
    """
    Try to find the new path for a track.
    Strategy:
      1. Use the basename of the existing filepath for an exact (case-insensitive) lookup.
      2. If multiple hits, prefer the one whose directory contains the artist or title name.
    Returns the new absolute path, or None if not found.
    """
    old_path = track.get("filepath") or ""
    old_basename = Path(old_path).name if old_path else ""

    candidates = file_index.get(old_basename.lower(), [])

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple files with the same filename — pick the best one
    title  = (track.get("title")  or "").lower()
    artist = (track.get("artist") or "").lower()

    def score(path: str) -> int:
        p = path.lower()
        return (artist in p) + (title in p)

    candidates_sorted = sorted(candidates, key=score, reverse=True)
    return candidates_sorted[0]


def main():
    parser = argparse.ArgumentParser(description="Re-path tracks in Supabase after moving music files.")
    parser.add_argument("--root", required=True, help="Root directory to search for audio files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to DB.")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Build the file index
    file_index = build_file_index(args.root)

    # 2. Fetch all tracks
    tracks = fetch_all_tracks(supabase)
    print(f"Fetched {len(tracks)} tracks from Supabase.\n")

    matched   = []
    unmatched = []
    unchanged = []

    for track in tracks:
        new_path = resolve_new_path(track, file_index)
        old_path = track.get("filepath") or ""

        if new_path is None:
            unmatched.append(track)
        elif new_path == old_path:
            unchanged.append(track)
        else:
            matched.append((track, new_path))

    # 3. Report
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Results:")
    print(f"  Already correct : {len(unchanged)}")
    print(f"  Will update     : {len(matched)}")
    print(f"  Not found       : {len(unmatched)}\n")

    if matched:
        print("Updates:")
        for track, new_path in matched:
            print(f"  [{track['trackid']}] {track.get('title', '?')} — {track.get('artist', '?')}")
            print(f"    OLD: {track.get('filepath') or '(none)'}")
            print(f"    NEW: {new_path}")

    if unmatched:
        print("\nNot found (no file matched):")
        for track in unmatched:
            print(f"  [{track['trackid']}] {track.get('title', '?')} — {track.get('artist', '?')}")
            print(f"    Old path: {track.get('filepath') or '(none)'}")

    if args.dry_run:
        print("\nDry run complete — no DB changes made.")
        return

    # 4. Write updates
    if not matched:
        print("Nothing to update.")
        return

    print(f"\nUpdating {len(matched)} tracks in Supabase...")
    errors = []
    for track, new_path in matched:
        try:
            supabase.table("tracks") \
                .update({"filepath": new_path}) \
                .eq("trackid", track["trackid"]) \
                .execute()
        except Exception as e:
            errors.append((track, str(e)))

    if errors:
        print(f"\n{len(errors)} update(s) failed:")
        for track, err in errors:
            print(f"  [{track['trackid']}] {track.get('title', '?')}: {err}")

    success = len(matched) - len(errors)
    print(f"\nDone. {success}/{len(matched)} tracks updated successfully.")


if __name__ == "__main__":
    main()
