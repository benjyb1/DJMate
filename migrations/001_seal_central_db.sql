-- ============================================================================
-- Migration 001: Seal the central DJMate DB
-- ============================================================================
-- Run this in the CENTRAL Supabase SQL Editor (cvermotfxamubejfnoje)
-- to lock down per-user tables that should never have been exposed here.
--
-- The central DB should only contain: profiles (user auth data).
-- All per-user music data belongs in each user's own Supabase project.
-- ============================================================================

-- ── Enable RLS on all per-user tables ──────────────────────────────────────
-- With no permissive policies, these tables become fully locked.

alter table if exists tracks             enable row level security;
alter table if exists track_labels       enable row level security;
alter table if exists track_features     enable row level security;
alter table if exists tag_corrections    enable row level security;
alter table if exists tag_track_feedback enable row level security;
alter table if exists tag_operations_log enable row level security;
alter table if exists suggested_playlists enable row level security;
alter table if exists playlists          enable row level security;
alter table if exists playlist_tracks    enable row level security;
alter table if exists crate_sessions     enable row level security;
alter table if exists crates             enable row level security;
alter table if exists crate_tracks       enable row level security;

-- ── Drop any existing permissive policies ──────────────────────────────────
-- (In case schema.sql was partially run on this DB before)
-- Wrapped in exception handlers so missing tables don't abort the block.

do $$
declare
  t text;
begin
  for t in
    select unnest(array[
      'tracks', 'track_labels', 'track_features', 'tag_corrections',
      'tag_track_feedback', 'tag_operations_log', 'suggested_playlists',
      'playlists', 'playlist_tracks', 'crate_sessions', 'crates', 'crate_tracks'
    ])
  loop
    begin
      execute format('drop policy if exists "anon full access" on %I', t);
    exception when undefined_table then
      null; -- table doesn't exist on this DB, that's fine
    end;
  end loop;
end;
$$;

-- ── Fix mutable search_path on functions (clears the other advisories) ─────
-- Note: ALTER FUNCTION IF EXISTS is not valid PG syntax, so we use a DO block.

do $$
begin
  alter function public.handle_new_user() set search_path = public;
exception when undefined_function then null;
end;
$$;

do $$
begin
  alter function public.update_updated_at() set search_path = public;
exception when undefined_function then null;
end;
$$;

do $$
begin
  alter function public.match_tracks(vector, float, int) set search_path = public, extensions;
exception when undefined_function then null;
end;
$$;
