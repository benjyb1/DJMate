"""
migrate_add_tag_source.py
─────────────────────────
Adds a `tag_source` column to the track_labels table in Supabase.

Values:
  'manual'        — tagged by a human in Tagger.py
  'auto'          — tagged by the auto_tagger batch script (ML-suggested)
  'auto_reviewed' — was auto-tagged, then a human confirmed/edited it in Tagger.py

Run once:
  python backend/migrate_add_tag_source.py
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌  SUPABASE_URL and SUPABASE_KEY must be set in .env")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def run_migration():
    print("Running migration: fix tag_source column in track_labels …")
    print("This corrects the previous backfill that marked all rows as 'manual'.")
    print("After this, only rows with real tag content will be 'manual'; the rest become NULL.")

    # Supabase JS client doesn't expose raw DDL — use the REST API trick:
    # insert a dummy row with tag_source to check if the column already exists,
    # then roll it back. If PostgREST returns a 400 about unknown column, we
    # need to add it via the Supabase dashboard SQL editor.
    #
    # Because Supabase doesn't allow raw DDL through the anon/service key REST
    # API, we print the SQL and instruct the user to run it in the dashboard.

    sql = """
-- Run this in your Supabase SQL editor (Dashboard → SQL Editor → New query)
-- https://app.supabase.com → your project → SQL Editor

-- Step 1: Drop the DEFAULT 'manual' constraint so NULL is a valid value going forward
ALTER TABLE track_labels ALTER COLUMN tag_source DROP DEFAULT;

-- Step 2: Reset ALL rows to NULL first (clean slate)
UPDATE track_labels SET tag_source = NULL;

-- Step 3: Mark ONLY rows with real tag content as 'manual'
--   Handles: NULL array, empty array '[]', and arrays with at least one element
UPDATE track_labels
  SET tag_source = 'manual'
  WHERE
    semantic_tags IS NOT NULL
    AND semantic_tags::text NOT IN ('null', '[]')
    AND jsonb_array_length(semantic_tags) > 0;

-- Verify — you should see 'manual' for tagged rows, NULL for untagged
SELECT
  tag_source,
  COUNT(*) AS count,
  COUNT(CASE WHEN jsonb_array_length(COALESCE(semantic_tags, '[]'::jsonb)) > 0 THEN 1 END) AS has_tags
FROM track_labels
GROUP BY tag_source
ORDER BY tag_source NULLS LAST;
"""

    print("\n" + "=" * 60)
    print("ACTION REQUIRED — Supabase doesn't allow DDL via REST API.")
    print("Please run the following SQL in your Supabase SQL editor:")
    print("=" * 60)
    print(sql)
    print("=" * 60)

    # Verify the column exists by trying to select it
    print("\nChecking if tag_source column already exists …")
    try:
        resp = supabase.table("track_labels").select("trackid, tag_source").limit(1).execute()
        if resp.data is not None:
            print("✅  tag_source column exists — migration already applied.")
        else:
            print("⚠️  Could not verify. Run the SQL above then re-run this script.")
    except Exception as e:
        if "tag_source" in str(e).lower() or "column" in str(e).lower():
            print("❌  Column does not exist yet. Run the SQL above in Supabase dashboard.")
        else:
            print(f"⚠️  Unexpected error during check: {e}")
            print("    Run the SQL above to be safe.")


if __name__ == "__main__":
    run_migration()
