-- ============================================================================
-- DJMate Database Schema
-- Paste this into your Supabase SQL Editor to set up a fresh project.
-- ============================================================================

-- ── Extensions ──────────────────────────────────────────────────────────────

create extension if not exists "vector" with schema "extensions";

-- ── Core tables ─────────────────────────────────────────────────────────────

-- tracks: one row per audio file in your library
create table if not exists tracks (
  trackid       text primary key,
  filepath      text,
  title         text not null,
  artist        text not null,
  album         text,
  bpm           real,
  key           text,
  duration      real,
  album_art_url text,
  embedding     vector(1536),
  x_coord       real,        -- precomputed UMAP 3D coordinate
  y_coord       real,
  z_coord       real,
  created_at    timestamptz default now()
);

-- track_labels: genre tags, vibe descriptors, energy per track
create table if not exists track_labels (
  trackid        text primary key references tracks(trackid) on delete cascade,
  semantic_tags  jsonb default '[]'::jsonb,   -- e.g. ["house","techno"]
  vibe           jsonb default '[]'::jsonb,   -- e.g. ["dark","hypnotic"]
  energy         real,                        -- 1-10 scale
  tag_source     text,                        -- 'auto', 'manual', 'auto_reviewed'
  created_at     timestamptz default now(),
  updated_at     timestamptz default now()
);

-- tag_corrections: audit log of user edits (online learning)
create table if not exists tag_corrections (
  id          bigint generated always as identity primary key,
  trackid     text references tracks(trackid) on delete cascade,
  field       text not null,       -- 'semantic_tags', 'vibe', or 'energy'
  old_value   text,
  new_value   text,
  created_at  timestamptz default now()
);

-- ── Playlist tables ─────────────────────────────────────────────────────────

create table if not exists playlists (
  id           text primary key,
  name         text not null,
  source       text,           -- 'rekordbox', 'manual', etc.
  source_path  text,
  parent_id    text references playlists(id) on delete set null,
  created_at   timestamptz default now()
);

create table if not exists playlist_tracks (
  id           bigint generated always as identity primary key,
  playlist_id  text not null references playlists(id) on delete cascade,
  trackid      text not null references tracks(trackid) on delete cascade,
  position     integer not null default 0
);

-- ── Crate tables (set-building sessions) ────────────────────────────────────

create table if not exists crate_sessions (
  id          text primary key,
  name        text not null default 'Untitled Session',
  created_at  timestamptz default now()
);

create table if not exists crates (
  id               text primary key,
  session_id       text not null references crate_sessions(id) on delete cascade,
  parent_crate_id  text references crates(id) on delete set null,
  label            text not null default '',
  position         integer default 0,
  avg_bpm          real,
  avg_energy       real,
  dominant_key     text,
  dominant_tags    jsonb default '[]'::jsonb,
  created_at       timestamptz default now()
);

create table if not exists crate_tracks (
  id        bigint generated always as identity primary key,
  crate_id  text not null references crates(id) on delete cascade,
  trackid   text not null references tracks(trackid) on delete cascade,
  position  integer not null default 0
);

-- ── Indexes ─────────────────────────────────────────────────────────────────

-- IVFFlat index for fast cosine-similarity vector search.
-- NOTE: This index requires at least ~100 rows with non-null embeddings
-- to build properly. If you have fewer tracks, create it after ingesting.
create index if not exists idx_tracks_embedding_cosine
  on tracks using ivfflat (embedding vector_cosine_ops)
  with (lists = 50);

create index if not exists idx_track_labels_trackid
  on track_labels (trackid);

create index if not exists idx_playlist_tracks_playlist_id
  on playlist_tracks (playlist_id);

create index if not exists idx_playlist_tracks_trackid
  on playlist_tracks (trackid);

create index if not exists idx_playlists_parent_id
  on playlists (parent_id);

create index if not exists idx_crate_tracks_crate_id
  on crate_tracks (crate_id);

create index if not exists idx_crates_session_id
  on crates (session_id);

create index if not exists idx_tag_corrections_trackid
  on tag_corrections (trackid);

-- ── RPC: match_tracks (vector similarity search) ────────────────────────────

create or replace function match_tracks(
  query_embedding vector(1536),
  match_threshold float default 0.3,
  match_count     int   default 20
)
returns table (
  trackid    text,
  title      text,
  artist     text,
  album      text,
  bpm        real,
  key        text,
  filepath   text,
  similarity float
)
language plpgsql
as $$
begin
  return query
    select
      t.trackid,
      t.title,
      t.artist,
      t.album,
      t.bpm,
      t.key,
      t.filepath,
      (1 - (t.embedding <=> query_embedding))::float as similarity
    from tracks t
    where t.embedding is not null
      and (1 - (t.embedding <=> query_embedding)) > match_threshold
    order by t.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- ── Row Level Security ──────────────────────────────────────────────────────
-- Each user owns their own Supabase project, so anon gets full access.

alter table tracks           enable row level security;
alter table track_labels     enable row level security;
alter table tag_corrections  enable row level security;
alter table playlists        enable row level security;
alter table playlist_tracks  enable row level security;
alter table crate_sessions   enable row level security;
alter table crates           enable row level security;
alter table crate_tracks     enable row level security;

-- Allow anon (and authenticated) full read/write on all tables
create policy "anon full access" on tracks          for all using (true) with check (true);
create policy "anon full access" on track_labels    for all using (true) with check (true);
create policy "anon full access" on tag_corrections for all using (true) with check (true);
create policy "anon full access" on playlists       for all using (true) with check (true);
create policy "anon full access" on playlist_tracks for all using (true) with check (true);
create policy "anon full access" on crate_sessions  for all using (true) with check (true);
create policy "anon full access" on crates          for all using (true) with check (true);
create policy "anon full access" on crate_tracks    for all using (true) with check (true);

-- ── Storage (manual step) ───────────────────────────────────────────────────
-- Supabase Storage buckets cannot be created via SQL. After running this
-- script, go to Storage in the Supabase dashboard and:
--
-- 1. Create a public bucket named "album-covers"
--    - Toggle "Public bucket" ON so cover art URLs are publicly accessible
--    - The upload script (scripts/upload_album_covers.py) writes here
--
-- 2. Create a public bucket named "audio" (optional, for streaming)
--    - Only needed if you upload audio files to Supabase instead of
--      serving them from local disk
