# Suggested Playlists Design

## Overview

Auto-generated playlist suggestions for DJs. Groups of 7-21 tracks that share a similar feeling, energy, and sonic character. Generated weekly, displayed in the playlist section with a refresh button to rotate through the pool.

## Algorithm: Seed-and-Expand with Minimum Seed Distance

### Execution Strategy

All embedding math happens in Python (NumPy). Load the full embedding matrix once (~794 tracks x 2560 dims, ~15 MB) and operate in memory. This avoids 10+ round-trip SQL queries and is fast enough for this data size. If the library grows past ~5000 tracks, add an HNSW index on `tracks.embedding` and shift to pgvector queries.

Check whether stored embeddings are already L2-normalised. If they are, cosine distance = `1 - dot_product` (cheaper). If not, normalise at load time.

### Step 1: Seed Selection

1. Load all track embeddings from `tracks.embedding` into a NumPy matrix. Normalise to unit vectors if not already.
2. Pick the first seed randomly.
3. For each subsequent seed:
   - Candidate must have cosine distance > **minimum seed distance** (tunable, start at ~0.4) from ALL existing seeds.
   - Pick randomly from valid candidates.
   - Remove selected seed from the candidate pool.
4. Stop when 10 seeds are selected, or no more valid candidates exist.
5. If fewer than 10 seeds found, reduce minimum seed distance by 0.05 and retry (floor at 0.25).

### Step 2: Radius-Based Expansion (Voronoi Assignment)

Rather than sequential first-claim, assign every track to its nearest seed (Voronoi partitioning), then filter:

1. For each track, compute cosine distance to all seeds. Assign it to the nearest seed.
2. Within each seed's Voronoi cell, keep only tracks within the **expansion radius** (tunable, start at ~0.28 cosine distance).
3. **BPM gate:** Normalise BPM into a canonical range (80-160) by halving or doubling as needed, then drop any track whose normalised BPM deviates more than 15 from the cluster's median normalised BPM.
4. **Size constraints:**
   - Minimum: 7 tracks (including seed). Discard playlists that fall below this after filtering.
   - Maximum: 21 tracks. If more qualify, keep the 21 closest to the seed.
5. If a playlist is discarded (under 7), its tracks remain unassigned.

### Step 3: LLM Naming

For each playlist:

1. Aggregate data from member tracks (skip tracks with null labels):
   - Most common semantic tags (top 3-5 by frequency)
   - Most common vibes (top 3-5)
   - Average energy (1-10 scale)
   - BPM range (min-max, using original BPM values)
2. Send to LLM (cheapest available provider):
   - Prompt: "You are naming DJ playlists. Given tracks with tags [X], vibes [Y], average energy Z/10, BPM range A-B, generate a short playlist name. Maximum 2 words. Be creative and evocative, not generic. Examples: 'Velvet Groove', 'Acid Cathedral', 'Neon Drift'. Reply with ONLY the name."
3. **Fallback:** If the LLM call fails, generate a name from the top semantic tag + energy descriptor (e.g., "Techno Burner", "Deep Ambient"). Never block generation on LLM availability.
4. Store the name with the playlist.

### Step 4: Storage

New table `suggested_playlists`:

```sql
CREATE TABLE suggested_playlists (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  seed_trackid INTEGER REFERENCES tracks(trackid),
  track_ids INTEGER[] NOT NULL,
  generated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_suggested_playlists_generated ON suggested_playlists(generated_at DESC);
```

- Each generation produces ~10 playlists (fewer if library density doesn't support it).
- Old generations are kept for history. The GET endpoint returns the most recent generation (not calendar-week dependent).
- Regeneration: weekly cron job, or manual trigger via API endpoint.
- Dangling track IDs are acceptable since playlists are regenerated weekly. Stale references are replaced on next generation.

### Step 5: API

New router: `suggested_playlist_router.py`, mounted at `/suggested-playlists` (separate from existing `/playlists` router).

**`POST /suggested-playlists/generate`** — Trigger playlist generation (admin/manual).
- Writes a new batch with current timestamp.
- Returns the generated playlists.

**`GET /suggested-playlists`** — Fetch most recent generation.
- Returns all playlists from the latest `generated_at` batch.
- Response schema per playlist:
  ```json
  {
    "id": 1,
    "name": "Velvet Groove",
    "seed_trackid": 42,
    "track_count": 14,
    "bpm_range": [118, 128],
    "tracks": [
      {"trackid": 42, "title": "...", "artist": "...", "album_art_url": "...", "bpm": 122},
      ...
    ],
    "generated_at": "2026-03-26T00:00:00Z"
  }
  ```
- Full track objects returned so the frontend can render cards without additional API calls.

### Step 6: Frontend

- Playlist section shows **3 playlists** at a time.
- Each playlist card shows:
  - 2-word LLM name
  - Track count
  - BPM range
  - Album art mosaic (first 4 track covers)
- **Refresh button** rotates to the next batch of 3 from the pool of ~10.
  - Simple offset cycling: 0-2, 3-5, 6-8, 9 (last batch wraps or shows fewer).
- Clicking a playlist expands it to show all tracks.
- Tracks are playable and selectable (same interactions as elsewhere in the app).

## Tunable Parameters

| Parameter | Starting Value | Purpose |
|---|---|---|
| Minimum seed distance | 0.4 cosine | Prevents overlapping playlists |
| Expansion radius | 0.28 cosine | How tight/loose each playlist is |
| BPM deviation max | 15 BPM (normalised) | Keeps playlists DJ-mixable |
| BPM canonical range | 80-160 | Half/double normalisation range |
| Target playlist count | 10 | Pool size per generation |
| Min playlist size | 7 | Discard threshold |
| Max playlist size | 21 | Cap per playlist |
| Seed distance floor | 0.25 | Minimum after retry reduction |

These will need tuning against the actual data distribution. Worth running a test generation and eyeballing the results before locking values in.

## Generation Flow (Pseudocode)

```
load all embeddings into numpy matrix
normalise to unit vectors (if not already)
load bpm for all tracks
normalise_bpm(bpm) = bpm/2 if bpm > 160 else bpm*2 if bpm < 80 else bpm

seeds = []
available = set(all_track_ids)

# Seed selection
while len(seeds) < 10:
    candidates = [t for t in available
                  if all(cosine_dist(t, s) > min_seed_dist for s in seeds)]
    if not candidates:
        min_seed_dist -= 0.05
        if min_seed_dist < 0.25: break
        continue
    seed = random.choice(candidates)
    seeds.append(seed)
    available.remove(seed)

# Voronoi assignment: each track -> nearest seed
assignments = {}  # seed_id -> [track_ids]
for track in all_tracks:
    nearest_seed = argmin(cosine_dist(track, s) for s in seeds)
    assignments.setdefault(nearest_seed, []).append(track)

# Filter and build playlists
playlists = []
for seed in seeds:
    cell = assignments.get(seed, [])
    # Radius filter
    cell = [t for t in cell if cosine_dist(seed, t) <= expansion_radius]
    # BPM gate (normalised)
    all_bpms = [normalise_bpm(bpm(t)) for t in cell + [seed]]
    med_bpm = median(all_bpms)
    cell = [t for t in cell if abs(normalise_bpm(bpm(t)) - med_bpm) <= 15]
    # Size constraints
    if len(cell) + 1 < 7:  # +1 for seed
        continue
    cell.sort(by=cosine_dist_to_seed)
    cell = cell[:20]  # 20 + seed = 21 max
    playlists.append([seed] + cell)

# Naming
for playlist in playlists:
    aggregate = get_tag_vibe_energy_bpm(playlist, skip_null_labels=True)
    name = llm_generate_name(aggregate) or fallback_name(aggregate)
    store(playlist, name)
```

## Out of Scope

- Track ordering within playlists (DJ handles this)
- User-created playlists (separate feature)
- Personalisation based on listening history (future enhancement)
- Cross-library playlists (single user's library only)
