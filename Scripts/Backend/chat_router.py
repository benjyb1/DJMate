"""
chat_router.py — FastAPI router that exposes the SemanticInterpreter
to the React frontend via two endpoints:

  POST /chat/interpret   → parse a natural-language query into params
  POST /chat/search      → run the full search and return enriched tracks

Mount in your main FastAPI app:
  from chat_router import router as chat_router
  app.include_router(chat_router, prefix="/chat", tags=["chat"])
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy singletons (initialised on first request) ─────────────────────────────
_interpreter = None
_db          = None


def _get_db():
    global _db
    if _db is None:
        from Backend.data.db_interface import DatabaseManager
        _db = DatabaseManager(enable_caching=True, pool_size=10)
    return _db


async def _get_interpreter():
    global _interpreter
    if _interpreter is None:
        from Backend.llm_interpreter import SemanticInterpreter
        db = _get_db()
        _interpreter = SemanticInterpreter(supabase_client=db.client)
        await _interpreter.initialize()
    return _interpreter


# ── Request / Response models ──────────────────────────────────────────────────

class InterpretRequest(BaseModel):
    query: str


class InterpretResponse(BaseModel):
    params: Dict[str, Any]


class SearchRequest(BaseModel):
    params: Dict[str, Any]


class TrackResult(BaseModel):
    trackid:          Optional[str]       = None
    title:            Optional[str]       = None
    artist:           Optional[str]       = None
    bpm:              Optional[float]     = None
    key:              Optional[str]       = None
    energy:           Optional[float]     = None
    filepath:         Optional[str]       = None
    semantic_tags:    Optional[List[str]] = []
    vibe_descriptors: Optional[List[str]] = []
    relevance_score:  Optional[float]     = None
    inferred:         Optional[bool]      = False


class SearchResponse(BaseModel):
    tracks:           List[TrackResult]
    relaxation_step:  int
    relaxation_label: str
    inferred_count:   int
    track_count:      int
    reasoning:        str
    confidence:       float
    model_used:       str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/interpret", response_model=InterpretResponse)
async def interpret(req: InterpretRequest):
    """
    Parse a free-text DJ query into structured search parameters.
    Mirrors: params = run_async(interpreter.interpret(query_input.strip()))
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        interp = await _get_interpreter()
        params = await interp.interpret(req.query.strip())
        return InterpretResponse(params=params)
    except Exception as e:
        logger.error(f"Interpret error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Run the full semantic search pipeline and return enriched tracks.
    Mirrors: tracks, meta = run_async(interpreter.search(params, db_manager=db))
    """
    try:
        interp = await _get_interpreter()
        db     = _get_db()

        tracks_raw, meta = await interp.search(req.params, db_manager=db)

        tracks = []
        for t in tracks_raw:
            tracks.append(TrackResult(
                trackid=          t.get("trackid") or t.get("id"),
                title=            t.get("title",   "Unknown"),
                artist=           t.get("artist",  "Unknown"),
                bpm=              t.get("bpm"),
                key=              t.get("key"),
                energy=           t.get("energy"),
                filepath=         t.get("filepath"),
                semantic_tags=    t.get("semantic_tags")    or [],
                vibe_descriptors= t.get("vibe_descriptors") or t.get("vibe") or [],
                relevance_score=  t.get("_relevance_score", 0.5),
                inferred=         t.get("_inferred", False),
            ))

        return SearchResponse(
            tracks=           tracks,
            relaxation_step=  meta.get("relaxation_step",  0),
            relaxation_label= meta.get("relaxation_label", ""),
            inferred_count=   meta.get("inferred_count",   0),
            track_count=      req.params.get("track_count", len(tracks)),
            reasoning=        req.params.get("reasoning",  ""),
            confidence=       req.params.get("confidence", 0.0),
            model_used=       req.params.get("model_used", "fallback"),
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Quick liveness check for the chat endpoints."""
    return {"status": "ok", "endpoints": ["/chat/interpret", "/chat/search"]}