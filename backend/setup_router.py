"""Setup router — serves the schema SQL for new-user onboarding."""
from fastapi import APIRouter

router = APIRouter()

SCHEMA_SQL = """
-- DJMate Schema — paste this into your Supabase SQL Editor

create extension if not exists "vector" with schema "extensions";

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
  audio_url     text,
  embedding     vector(1536),
  x_coord       real,
  y_coord       real,
  z_coord       real,
  created_at    timestamptz default now()
);

create table if not exists track_labels (
  trackid        text primary key references tracks(trackid) on delete cascade,
  semantic_tags  jsonb default '[]'::jsonb,
  vibe           jsonb default '[]'::jsonb,
  energy         real,
  tag_source     text,
  created_at     timestamptz default now(),
  updated_at     timestamptz default now()
);

create table if not exists track_features (
  trackid    text primary key references tracks(trackid) on delete cascade,
  mfcc       float8[],
  created_at timestamptz default now()
);

create table if not exists tag_corrections (
  id          bigint generated always as identity primary key,
  trackid     text references tracks(trackid) on delete cascade,
  field       text not null,
  old_value   text,
  new_value   text,
  created_at  timestamptz default now()
);

create table if not exists playlists (
  id           text primary key,
  name         text not null,
  source       text,
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

-- Tag operations log
create table if not exists tag_operations_log (
  id          bigint generated always as identity primary key,
  operation   text not null,
  params      jsonb,
  result      jsonb,
  created_at  timestamptz default now()
);

-- Tag track feedback
create table if not exists tag_track_feedback (
  id          bigint generated always as identity primary key,
  trackid     text references tracks(trackid) on delete cascade,
  field       text,
  old_value   text,
  new_value   text,
  created_at  timestamptz default now()
);

-- RLS: anon full access (each user owns their own project)
alter table tracks           enable row level security;
alter table track_labels     enable row level security;
alter table track_features   enable row level security;
alter table tag_corrections  enable row level security;
alter table playlists        enable row level security;
alter table playlist_tracks  enable row level security;
alter table crate_sessions   enable row level security;
alter table crates           enable row level security;
alter table crate_tracks     enable row level security;
alter table tag_operations_log enable row level security;
alter table tag_track_feedback enable row level security;

create policy "anon full access" on tracks          for all using (true) with check (true);
create policy "anon full access" on track_labels    for all using (true) with check (true);
create policy "anon full access" on track_features  for all using (true) with check (true);
create policy "anon full access" on tag_corrections for all using (true) with check (true);
create policy "anon full access" on playlists       for all using (true) with check (true);
create policy "anon full access" on playlist_tracks for all using (true) with check (true);
create policy "anon full access" on crate_sessions  for all using (true) with check (true);
create policy "anon full access" on crates          for all using (true) with check (true);
create policy "anon full access" on crate_tracks    for all using (true) with check (true);
create policy "anon full access" on tag_operations_log for all using (true) with check (true);
create policy "anon full access" on tag_track_feedback for all using (true) with check (true);

-- Indexes
create index if not exists idx_track_labels_trackid on track_labels (trackid);
create index if not exists idx_playlist_tracks_playlist_id on playlist_tracks (playlist_id);
create index if not exists idx_playlist_tracks_trackid on playlist_tracks (trackid);
create index if not exists idx_playlists_parent_id on playlists (parent_id);
create index if not exists idx_crate_tracks_crate_id on crate_tracks (crate_id);
create index if not exists idx_crates_session_id on crates (session_id);
create index if not exists idx_tag_corrections_trackid on tag_corrections (trackid);

-- RPC: match_tracks (vector similarity search)
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
"""


@router.get("/schema-sql")
async def get_schema_sql():
    """Return the SQL needed to set up a fresh Supabase project for DJMate."""
    return {"sql": SCHEMA_SQL}
