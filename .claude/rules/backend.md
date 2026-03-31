---
paths:
  - "backend/**/*.py"
  - "scripts/**/*.py"
---

# Backend Rules

## Stack
- FastAPI with async/await throughout
- Supabase (Postgres + asyncpg for direct pool connections)
- Python 3.11+ type hints

## LLM Architecture
- Tool-calling with 5 tools: `set_genre`, `set_vibe`, `set_energy`, `set_bpm`, `set_track_count`
- Genre and vibe are INDEPENDENT axes — adjectives → vibe, genre names → genre
- Energy scale: 1–10 integer (NOT 0.0–1.0)
- Provider fallback order: Groq → Gemini → OpenAI → Mistral
- Fallback: JSON prompt → keyword extraction

## Scoring Weights
- Vibe-only: tag=10%, vibe=65%
- Genre-only: tag=60%, vibe=10%
- Mixed (vibe-dominant): tag=30%, vibe=50%
- Mixed (genre-dominant): tag=45%, vibe=35%

## Key Files
- `backend/llm_interpreter.py` — SemanticInterpreter
- `backend/chat_router.py` — `/chat/interpret` + `/chat/search`
- `backend/data/db_interface.py` — DatabaseManager (Supabase + asyncpg)
