-- Add IVFFlat index for fast vector similarity search
-- Run in Supabase SQL editor

CREATE INDEX IF NOT EXISTS idx_tracks_embedding_cosine
ON tracks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_track_labels_trackid
ON track_labels (trackid);

CREATE INDEX IF NOT EXISTS idx_playlist_tracks_playlist_id
ON playlist_tracks (playlist_id);

CREATE INDEX IF NOT EXISTS idx_playlist_tracks_trackid
ON playlist_tracks (trackid);

CREATE INDEX IF NOT EXISTS idx_playlists_parent_id
ON playlists (parent_id);
