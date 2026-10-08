"""Next Track: a rekordbox companion that suggests where to go next.

Everything here is local-first. The library index lives on disk under
``paths.data_dir()``, audio analysis runs in an isolated Essentia subprocess,
and "now playing" comes from rekordbox's own play history. Nothing touches
Supabase, so the feature works offline at a gig.
"""
