# DJMate

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi)
![Essentia](https://img.shields.io/badge/Essentia-TensorFlow-FF6F00?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-kNN%20%2B%20CV-F7931E?logo=scikitlearn)
![Supabase](https://img.shields.io/badge/Supabase-Postgres%20%2B%20pgvector-3ECF8E?logo=supabase)
![React](https://img.shields.io/badge/React-18%20%2B%20Three.js-61DAFB?logo=react)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

DJMate is a music recommendation system for DJs. It turns a personal track library into a space you can search and navigate by feel, built on per-track audio embeddings, a stack of transfer-learning classifiers, and an online learning loop that keeps sharpening its labels from human corrections.

A React and Three.js frontend renders the whole library as a 3D map. A natural-language chat panel lets you pull tracks by intent ("dark hypnotic techno", "take it higher") instead of by filename. It began as a King's Coding Club project and turned into the tool I actually use to plan sets.

---

## Screenshots

The 3D library map. Every node is a track, positioned by audio similarity.
<img src="screenshots/3d-overview.png" width="800">

Ask for tracks like one you're already on.
<img src="screenshots/3d-zoomed.png" width="800">

Or describe a vibe and let it steer the set.

<p float="left">
  <img src="screenshots/chat-vibe.png" width="400">
  <img src="screenshots/chat-find.png" width="400">
</p>

The tagging backend, where the online learning loop suggests labels and learns from my corrections.
<img src="screenshots/tagger.png" width="800">

---

## The machine learning, in one screen

Everything here runs off a single signal: a 1536-dimension audio embedding per track (768-dim Discogs-EffNet activations, mean- and std-pooled across frames). The interesting parts are what gets layered on top of it.

| Component | Technique | What it does |
|---|---|---|
| **Embeddings** | Discogs-EffNet (Essentia + TensorFlow) | One 1536-dim vector per track (mean + std pooling), computed from the raw audio. Stored in Postgres with pgvector. |
| **Similarity search** | Cosine distance (pgvector + scikit-learn) | "Find tracks like this", nearest-neighbour retrieval, the foundation everything else builds on. |
| **Library understanding** | Pretrained transfer-learning heads | Mood (party, aggressive, relaxed) and danceability on EffNet, plus a DEAM valence/arousal regressor on MusiCNN embeddings. |
| **Energy model** | 5-fold cross-validated regression | Fits audio features to my own 1 to 10 energy ratings, compares feature sets by MAE, keeps only what actually predicts my ear. |
| **Auto-tagging** | Weighted k-NN (k = 10) | Labels an untagged track from its 10 nearest neighbours, similarity-weighted, with confidence thresholds and cascade protection. |
| **Online learning** | Embedding-space correction propagation | Learns from my tag corrections, clusters them, and propagates confident patterns to nearby tracks with recency-weighted confidence. |
| **Set building** | Clustering + harmonic rules | Seed-and-expand clustering for playlist suggestions; Camelot-wheel key matching and energy curves for sequencing a crate. |
| **Query layer** | LLM tool-calling | Turns a natural-language query into genre, vibe, energy, BPM and count, then routes it to the right search. |

The energy model is the one component with a proper held-out evaluation (5-fold CV against manual ratings, below). Auto-tagging, playlist coherence and search relevance are judged by ear rather than a held-out metric — worth knowing before assuming precision/recall numbers exist for them.

### Audio embeddings

Each track is run through the Discogs-EffNet model (Essentia on TensorFlow) at 16 kHz mono. Per-frame activations are mean- and std-pooled into a 1536-dimension embedding that captures timbre and texture far better than BPM and key alone. Vectors are L2-normalised and stored in a `pgvector` column, so similarity is just cosine distance. The live API retrieves neighbours through a Postgres `match_tracks` RPC; the offline tools use scikit-learn for the same maths in memory.

### Library understanding through transfer learning

On top of the embeddings sit a set of pretrained Essentia heads, applied per frame and averaged: danceability and three mood axes (party, aggressive, relaxed) on the EffNet backbone, plus a DEAM valence/arousal regressor running on MusiCNN embeddings. This is classic transfer learning: a heavy pretrained backbone does the listening, light task heads read off the qualities I care about.

### A calibrated energy model

Energy is the single most useful number for building a set, and no off-the-shelf head predicts it well, so I fit my own. `calibrate_energy.py` takes the EffNet heads, the DEAM arousal/valence scores and BPM as candidate features, then runs 5-fold cross-validation across feature subsets, scoring each by MAE, correlation, and how often it lands within one star of my manual rating. It keeps the subset that genuinely predicts my ear and saves the fitted coefficients to `energy_model.json`.

The honest findings are baked into the code: on an all-dance library the danceability head saturates near 1.0 for every track, so it carries no signal and gets dropped; the variance lives in the aggressive and relaxed moods, with arousal pulling its weight when available. A non-uniform mapping from the 1 to 10 scale onto stars keeps the low end strict, so dub-tempo material correctly lands at one star.

### Online learning from corrections

`online_learner.py` is the part that improves with use: online, human-in-the-loop learning that propagates confident label corrections across the embedding graph.

1. **Capture.** Every manual tag correction is logged with its track and timestamp.
2. **Cluster.** For each correction, the system finds the 20 nearest tracks by embedding cosine similarity and checks how many share the old, wrong value. If agreement clears 50%, that becomes a correction pattern, weighted by agreement and an exponential recency decay (30-day half-life). Overlapping patterns (cosine ≥ 0.90) are merged.
3. **Propagate.** Patterns above a 0.70 confidence threshold are applied to auto-tagged tracks within the pattern's embedding radius.

It only ever rewrites `auto` labels, never `manual` or `auto_reviewed` ones, and promotes what it touches to `auto_reviewed`. So a handful of corrections quietly fixes a whole neighbourhood, and human-verified work is never clobbered.

### Recommendation and set construction

A natural-language query first hits an LLM intent classifier that exposes five tools (`set_genre`, `set_vibe`, `set_energy`, `set_bpm`, `set_track_count`), treating genre and vibe as independent axes. From there:

- **find_similar** runs a pgvector nearest-neighbour search off a fuzzy title or artist match.
- **vibe / genre search** scores tracks against a canonical tag vocabulary, then walks a progressive relaxation ladder: exact tags first, dropping low-weight tags, then vibes, then falling back to embedding similarity alone, so a query never dead-ends on zero results.
- **playlist suggestions** come from seed-and-expand clustering over the EffNet embeddings, producing groups of 7 to 21 sonically coherent tracks with LLM-generated names.
- **crate sequencing** orders tracks by Camelot-wheel harmonic compatibility and a target energy curve, then exports straight back to Rekordbox XML.

The LLM layer is a single OpenAI-compatible client multiplexed across Groq (primary, llama-3.3-70b), Gemini (2.5-flash) and OpenAI, with automatic failover and a keyword-extraction fallback if every provider is down.

### The 3D map

Track positions come from a UMAP projection of the 1536-dim embeddings down to three dimensions (`n_components=3`, `random_state=42`), computed offline and cached in the database. The frontend reads the precomputed coordinates and renders each track as a node textured with its album art, so the camera can fly to any track and trace similarity edges to its neighbours with no runtime dimensionality reduction.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         React Frontend                         │
│      Three.js 3D map  ·  DJ chat panel  ·  Streamlit tagger    │
└────────────────────────────┬──────────────────────────────────┘
                             │ HTTP (FastAPI, async)
┌────────────────────────────▼──────────────────────────────────┐
│                        FastAPI Backend                         │
│   intent classifier (LLM tool-calling)                         │
│   chat / search · playlists · crates · tags · ingest           │
│   online_learner  →  correction propagation                    │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                Supabase (Postgres + pgvector)                  │
│   tracks (1536-dim embedding, UMAP x/y/z)                      │
│   track_labels (tags, vibe, energy, tag_source provenance)     │
└────────────────────────────▲──────────────────────────────────┘
                             │ writes embeddings, scores, coords
┌────────────────────────────┴──────────────────────────────────┐
│              Offline ML pipeline (scripts/, models/)           │
│   ingest → EffNet embeddings → mood/energy/arousal scoring     │
│   → energy calibration (5-fold CV) → UMAP projection           │
└───────────────────────────────────────────────────────────────┘
```

The split matters: heavy model inference runs offline in batch and writes its results to the database, so the API stays light enough for a Render free-tier instance and every request is a fast vector lookup rather than a model forward pass.

---

## Tech stack

- **ML / audio:** Essentia (TensorFlow), Discogs-EffNet and MusiCNN embeddings, scikit-learn (k-NN, cosine, cross-validation), NumPy, librosa, umap-learn
- **Backend:** FastAPI (async), Supabase (Postgres + pgvector), asyncpg, Redis caching
- **LLM:** OpenAI-compatible client across Groq, Gemini and OpenAI with failover
- **Frontend:** React 18, Three.js, Vite
- **Tagging UI:** Streamlit
- **DJ integration:** Rekordbox XML import/export, Camelot-wheel harmonic mixing

---

## Project layout

```
DJMate/
├── main.py                              # FastAPI entry point
├── backend/
│   ├── llm_interpreter.py               # LLM intent classifier + tag scorer
│   ├── chat_router.py                   # /chat/* search routes
│   ├── auto_tagger.py                   # batch weighted-kNN auto-tagger
│   ├── online_learner.py               # online correction-propagation loop
│   ├── Tagger.py                        # Streamlit tagging UI ("AI Brain")
│   ├── suggested_playlist_generator.py  # seed-and-expand embedding clustering
│   ├── crate_generator.py               # Camelot + energy-curve sequencing
│   ├── playlist_router.py               # playlist CRUD + Rekordbox import/export
│   ├── rekordbox_parser.py              # Rekordbox XML parsing
│   └── data/
│       ├── db_interface.py              # DatabaseManager (Supabase + asyncpg)
│       └── embeddings.py                # shared embedding-matrix helpers
├── scripts/
│   ├── ingest_music.py                  # scan audio → metadata + EffNet embeddings
│   ├── compute_effnet_energy.py         # EffNet-head energy scoring
│   ├── score_arousal_batch.py           # DEAM valence/arousal scoring
│   ├── score_energy_batch.py            # batch energy over a library
│   ├── calibrate_energy.py              # 5-fold CV energy model fit
│   ├── 3d_coordinator.py                # UMAP 3D projection
│   └── write_rekordbox_stars.py         # push energy stars back to Rekordbox
├── models/                              # Essentia EffNet / MusiCNN heads
├── configuration/
│   ├── requirements.txt                 # backend (lean, for Render)
│   └── requirements-scripts.txt         # offline ML extras
└── Frontend/                            # React + Three.js client
```

---

## Quick start

Full walkthrough, including Supabase setup, Essentia install and the embedding-model download, is in [SETUP.md](SETUP.md). The short version:

```bash
git clone https://github.com/benjyb1/DJMate.git
cd DJMate

# backend API
pip install -r configuration/requirements.txt
cp .env.example .env            # fill in SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY
uvicorn main:app --reload --port 8000

# offline ML scripts (audio analysis, embeddings, UMAP)
pip install -r configuration/requirements-scripts.txt

# frontend
cd Frontend && npm install && npm run dev

# tagging UI
streamlit run backend/Tagger.py
```

Ingest a library, then let the auto-tagger label it:

```bash
python scripts/ingest_music.py /path/to/your/music
python backend/auto_tagger.py --dry-run          # preview
python backend/auto_tagger.py --min-sim 0.70     # write
```

---

## Database schema

```sql
CREATE TABLE tracks (
    trackid   SERIAL PRIMARY KEY,
    filepath  TEXT NOT NULL,
    title     TEXT,
    artist    TEXT,
    bpm       FLOAT,
    key       TEXT,
    embedding VECTOR(1536),   -- Discogs-EffNet audio embedding (mean + std pooling)
    x_coord   FLOAT,          -- precomputed UMAP position
    y_coord   FLOAT,
    z_coord   FLOAT
);

CREATE TABLE track_labels (
    trackid       INTEGER REFERENCES tracks(trackid),
    semantic_tags JSONB,      -- e.g. ["Techno", "Acid House"]
    vibe          JSONB,      -- e.g. ["Dark", "Hypnotic", "Driving"]
    energy        INTEGER,    -- 1-10 scale
    tag_source    TEXT        -- 'manual' | 'auto' | 'auto_reviewed'
);
```

---

## License

MIT, see [LICENSE](LICENSE).

---

*Built for DJs who care about flow, and as a study in making audio embeddings useful end to end.*
