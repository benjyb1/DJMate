# Database Migrations

Run these SQL files in order in your Supabase SQL editor.

## 001_add_pgvector_index.sql
Adds IVFFlat index on tracks.embedding for fast cosine similarity search.
Also adds indexes on foreign keys for playlist and label queries.
