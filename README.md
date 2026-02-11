AI-Assisted DJ Music Curation System

[!Python](https://www.python.org/downloads/)
[!License: MIT](https://opensource.org/licenses/MIT)
[!Streamlit](https://streamlit.io)
[!Supabase](https://supabase.com)

An intelligent music curation system that helps DJs discover, sequence, and explore tracks from their personal libraries using AI-powered semantic understanding, audio embeddings, and harmonic mixing principles.

🎯 Vision

Traditional genre-based music systems fail to capture the nuanced relationships that matter to DJs. This project builds a semantic music space grounded in real DJ practice: vibe, energy, mood, groove, harmonic compatibility, and contextual similarity.

Natural Language Queries (Roadmap):
"Give me a moody minimal electro playlist"

"Suggest tracks similar to this one, but slightly deeper"

"Build a late-night set around this vibe at 124–128 BPM"

"I've played these tracks already — what should I play next?"

🏗️ Architecture

Core Components

graph TB
    A[Audio Files] --> B[Feature Extraction]
    B --> C[Embedding Generation]
    C --> D[FAISS Index]
    
    E[Human Tagging UI] --> F[Semantic Labels]
    F --> G[PostgreSQL Database]
    
    H[3D Visualization] --> I[UMAP Projection]
    D --> I
    G --> I
    
    J[Recommendation Engine] --> K[Confidence Scoring]
    D --> J
    G --> J
    
    L[LLM Query Parser] --> J
    J --> H

Database Schema

-- Immutable track metadata
CREATE TABLE tracks (
    trackid UUID PRIMARY KEY,
    filepath TEXT NOT NULL,
    title TEXT,
    artist TEXT,
    album TEXT,
    bpm FLOAT,
    key TEXT,
    embedding VECTOR(512)  -- High-dimensional audio features
);

-- Mutable semantic interpretations
CREATE TABLE track_labels (
    trackid UUID REFERENCES tracks(trackid),
    semantic_tags JSONB,     -- ["minimal", "deep", "hypnotic"]
    energy INTEGER,          -- 1-10 scale
    vibe JSONB,             -- ["dark", "driving", "introspective"]
    context JSONB,          -- ["late_night", "peak_time", "warm_up"]
    created_at TIMESTAMP DEFAULT NOW()
);

🚀 Quick Start

Prerequisites

# System requirements
Python 3.8+
Node.js 16+ (for VS Code frontend)
PostgreSQL 13+

Installation

# Clone repository
git clone https://github.com/benjyb/djmate.git
cd djmate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

Environment Setup

# Copy environment template
cp .env.example .env

# Configure your settings
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MUSIC_LIBRARY_PATH=/path/to/your/music
OPENAI_API_KEY=your_openai_key  # Optional, for LLM features

Database Setup

# Initialize database schema
python scripts/init_database.py

# Extract audio features from your library
python scripts/extract_features.py --library-path /path/to/music

# Build FAISS index
python scripts/build_faiss_index.py

🎛️ Usage

1. Semantic Tagging Interface

# Launch human-in-the-loop tagging UI
streamlit run src/tagging_interface.py

Features:
✅ Random sampling of unlabeled tracks

✅ Local audio playback (no uploads)

✅ Freeform semantic tagging

✅ Energy and vibe classification

✅ Progress tracking

✅ Tag reuse and autocomplete

2. 3D Music Space Visualization

# Launch VS Code extension frontend
cd frontend
npm run dev

Features:
✅ Interactive 3D point cloud of your music library

✅ UMAP dimensionality reduction

✅ Album artwork as node textures

🚧 Click-to-query similar tracks

🚧 Semantic filtering

🚧 Recommendation path visualization

3. Recommendation Engine

from src.recommendation_engine import DJRecommendationEngine

# Initialize engine
engine = DJRecommendationEngine()

# Get similar tracks
similar = engine.get_similar_tracks(
    track_id="uuid-here",
    weights={
        'embedding': 0.6,
        'semantic': 0.4, 
        'bpm': 0.3,
        'key': 0.2
    },
    limit=10
)

# Semantic query
results = engine.semantic_search(
    tags=["minimal", "deep"],
    energy_range=(3, 7),
    bpm_range=(120, 128),
    exclude_tags=["aggressive"]
)

🧠 Core Principles

1. No Rigid Genres
Tracks exist in a continuous semantic space. A single track can simultaneously be:
minimal + electro + tech house + deep + hypnotic

2. DJ-Centric Constraints
Recommendations respect real mixing requirements:
BPM Proximity: Mixable tempo ranges

Harmonic Compatibility: Camelot wheel relationships

Energy Flow: Coherent set progression

3. Human-AI Collaboration
DJs provide semantic labels through lightweight UI. AI handles:
Audio feature extraction

Similarity computation

Constraint satisfaction

Query parsing

4. Multi-Signal Fusion
Confidence-weighted scoring combines:

confidence = (
    w_embedding * embedding_similarity +
    w_key * harmonic_compatibility +
    w_bpm * tempo_proximity +
    w_semantic * tag_overlap +
    w_energy * energy_compatibility
)

🔬 Technical Deep Dive

Audio Feature Extraction

# Extract high-dimensional embeddings
from src.audio_processing import AudioFeatureExtractor

extractor = AudioFeatureExtractor(model='openl3')  # or 'musicnn', 'vggish'
embedding = extractor.extract_features(audio_path)

Supported models:
OpenL3: General audio representations

MusicNN: Music-specific features  

VGGish: YouTube-8M pretrained

Custom: Fine-tuned on electronic music

Harmonic Mixing Logic

# Camelot wheel compatibility
HARMONIC_RELATIONSHIPS = {
    'same_key': 1.0,
    'relative_major_minor': 0.9,
    'perfect_fifth': 0.8,
    'adjacent_camelot': 0.7,
    'compatible': 0.5,
    'clash': 0.1
}

def compute_key_compatibility(key1: str, key2: str) -> float:
    return HARMONIC_RELATIONSHIPS.get(
        get_relationship(key1, key2), 
        0.1
    )

Semantic Embedding Space

# Project high-dimensional embeddings to 3D
from umap import UMAP

reducer = UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine'
)

coordinates_3d = reducer.fit_transform(audio_embeddings)

🗂️ Project Structure

ai-dj-curation/
├── src/
│   ├── audio_processing/          # Feature extraction
│   ├── database/                  # Supabase integration
│   ├── recommendation_engine/     # Core ML logic
│   ├── semantic_processing/       # NLP and tagging
│   └── visualization/            # UMAP and plotting
├── frontend/                     # VS Code extension
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── scripts/                      # Setup and maintenance
├── tests/                        # Unit and integration tests
├── docs/                         # Documentation
├── requirements.txt
└── README.md

🎵 Data Flow

Ingestion: Scan music library, extract metadata

Feature Extraction: Generate audio embeddings

Human Annotation: Tag tracks via Streamlit UI

Index Building: Create FAISS similarity index

Visualization: Project to 3D space via UMAP

Query Processing: Parse natural language → structured query

Recommendation: Multi-signal fusion → ranked results

Feedback Loop: User interactions improve future recommendations

🚧 Roadmap

Phase 1: Core Integration (Current)
Database schema and ingestion pipeline

Streamlit tagging interface  

FAISS similarity search

3D visualization frontend

Frontend ↔ Backend API integration

Basic recommendation engine

Phase 2: Advanced Semantics
Hierarchical/contextual tagging

LLM query parsing

Set sequence building

Confidence calibration

Phase 3: DJ-Specific Features  
Live set integration

Energy flow modeling

Recommendation explanations

A/B testing framework

Phase 4: Production
Real-time performance optimization

Mobile companion app

Cloud deployment

Multi-user support

📊 Evaluation Metrics

Recommendation Quality
Precision@K: Relevant tracks in top-K results

Semantic Coherence: Tag overlap in recommendations

Harmonic Accuracy: Key compatibility scores

Tempo Consistency: BPM distribution analysis

System Performance
Query Latency: Response time for similarity search

Index Build Time: FAISS construction performance  

Memory Usage: Embedding storage efficiency

Throughput: Concurrent query handling

User Experience
Tagging Velocity: Labels per minute in UI

Discovery Rate: New tracks surfaced per session

Query Success: Natural language understanding accuracy

🤝 Contributing

We welcome contributions from the DJ and ML communities!

Development Setup

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
isort src/

# Type checking  
mypy src/

Contribution Areas

Audio ML: Improve embedding models for electronic music

UI/UX: Enhance tagging interface and visualization

DJ Workflow: Add real-world mixing features

Performance: Optimize similarity search and indexing

Documentation: Examples, tutorials, best practices

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

OpenL3: Audio embedding foundation

FAISS: Efficient similarity search  

Supabase: Database and backend infrastructure

Streamlit: Rapid prototyping framework

UMAP: Dimensionality reduction and visualization

Mixed In Key: Harmonic mixing inspiration

📞 Contact

Issues: GitHub Issues

Discussions: GitHub Discussions

Email: your.email@example.com




Built with ❤️ for the DJ community. This is not a consumer playlist generator — it's a tool for DJs who care about musical coherence, flow, and intent.
