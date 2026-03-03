"""
remove_duplicates.py
─────────────────────
Finds and removes duplicate tracks from the DB.

Two duplication signals (both checked):
  1. Same filepath  — the exact same file was ingested twice
  2. Same title + artist — same song from different file paths

Within each duplicate group the "best" row is kept:
  - Prefer tracks that have an embedding (needed for search)
  - Then prefer tracks that have album_art_url
  - Then keep the one with the lowest trackid (ingested first)

Orphaned track_labels rows for deleted tracks are also cleaned up.

Usage:
  python backend/remove_duplicates.py           # dry-run (preview only)
  python backend/remove_duplicates.py --write   # delete from DB
"""

import os
import sys
import argparse
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


def get_supabase():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌  SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


def score_track(t: dict) -> tuple:
    """
    Higher score = better candidate to keep.
    Returns a tuple so we can sort: (has_embedding, has_album_art, -trackid)
    Negating trackid means lower trackid wins ties (ingested first).
    """
    has_embedding  = 1 if t.get("embedding")      else 0
    has_album_art  = 1 if t.get("album_art_url")  else 0
    return (has_embedding, has_album_art, -t["trackid"])


def find_duplicate_groups(tracks: list[dict]) -> list[list[dict]]:
    """
    Returns a list of groups, where each group is a list of tracks
    that are duplicates of each other.  Groups with only 1 member are excluded.
    """
    # Group by filepath first (most definitive)
    by_filepath = defaultdict(list)
    for t in tracks:
        fp = (t.get("filepath") or "").strip().lower()
        if fp:
            by_filepath[fp].append(t)

    # Group by normalised title + artist
    by_title_artist = defaultdict(list)
    for t in tracks:
        title  = (t.get("title")  or "").strip().lower()
        artist = (t.get("artist") or "").strip().lower()
        if title:  # don't group tracks with no title
            by_title_artist[(title, artist)].append(t)

    # Group by normalised title alone (catches same song with different artist metadata)
    by_title = defaultdict(list)
    for t in tracks:
        title = (t.get("title") or "").strip().lower()
        if title:
            by_title[title].append(t)

    # Merge the two groupings: if two tracks share a filepath OR a title+artist
    # they belong to the same duplicate group.  Use Union-Find for correctness.
    trackid_to_idx = {t["trackid"]: i for i, t in enumerate(tracks)}
    parent = list(range(len(tracks)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for group in by_filepath.values():
        if len(group) > 1:
            for t in group[1:]:
                union(trackid_to_idx[group[0]["trackid"]],
                      trackid_to_idx[t["trackid"]])

    for group in by_title_artist.values():
        if len(group) > 1:
            for t in group[1:]:
                union(trackid_to_idx[group[0]["trackid"]],
                      trackid_to_idx[t["trackid"]])

    for group in by_title.values():
        if len(group) > 1:
            for t in group[1:]:
                union(trackid_to_idx[group[0]["trackid"]],
                      trackid_to_idx[t["trackid"]])

    # Collect into root → members map
    root_to_members = defaultdict(list)
    for t in tracks:
        root_to_members[find(trackid_to_idx[t["trackid"]])].append(t)

    return [members for members in root_to_members.values() if len(members) > 1]


def main():
    parser = argparse.ArgumentParser(
        description="Find and remove duplicate tracks from the DB."
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Delete duplicates from the DB.  Without this flag the script is a dry-run."
    )
    args = parser.parse_args()

    dry_run = not args.write
    if dry_run:
        print("🔍  DRY RUN — no changes will be made.  Pass --write to delete.\n")

    supabase = get_supabase()

    # ── Load all tracks (no embedding vector — too large; just metadata) ───────
    print("Loading tracks …")
    resp = supabase.table("tracks") \
        .select("trackid, title, artist, filepath, album_art_url, embedding") \
        .execute()
    tracks = resp.data or []
    # Mark embedding presence as bool to avoid pulling huge vectors into memory
    for t in tracks:
        t["embedding"] = bool(t.get("embedding"))
    print(f"  {len(tracks)} tracks loaded\n")

    # ── Find duplicate groups ──────────────────────────────────────────────────
    groups = find_duplicate_groups(tracks)

    if not groups:
        print("✅  No duplicates found.")
        return

    total_to_delete = sum(len(g) - 1 for g in groups)
    print(f"Found {len(groups)} duplicate group(s) — {total_to_delete} track(s) will be removed.\n")

    to_delete_ids = []

    for i, group in enumerate(groups, 1):
        # Sort by score descending — first entry is the keeper
        group_sorted = sorted(group, key=score_track, reverse=True)
        keeper  = group_sorted[0]
        victims = group_sorted[1:]

        print(f"  Group {i}:  KEEP  [{keeper['trackid']}] "
              f"{keeper['artist'] or 'Unknown'} — {keeper['title'] or '(no title)'}  "
              f"(embedding={'✓' if keeper['embedding'] else '✗'}  "
              f"art={'✓' if keeper['album_art_url'] else '✗'})")
        for v in victims:
            print(f"           DELETE [{v['trackid']}] "
                  f"{v['artist'] or 'Unknown'} — {v['title'] or '(no title)'}  "
                  f"(embedding={'✓' if v['embedding'] else '✗'}  "
                  f"art={'✓' if v['album_art_url'] else '✗'})")
            to_delete_ids.append(v["trackid"])
        print()

    if dry_run:
        print(f"─── Dry run complete — {total_to_delete} track(s) would be deleted. ───")
        print("    Run with --write to apply.")
        return

    # ── Delete victims ─────────────────────────────────────────────────────────
    print(f"Deleting {len(to_delete_ids)} duplicate track(s) …")
    deleted = 0
    failed  = 0

    for tid in to_delete_ids:
        try:
            # Remove track_labels first (foreign key safety)
            supabase.table("track_labels").delete().eq("trackid", tid).execute()
            # Remove the track itself
            supabase.table("tracks").delete().eq("trackid", tid).execute()
            deleted += 1
        except Exception as e:
            print(f"  ⚠️  Failed to delete trackid {tid}: {e}")
            failed += 1

    print(f"\n✅  Done.  Deleted: {deleted}  Failed: {failed}")


if __name__ == "__main__":
    main()
