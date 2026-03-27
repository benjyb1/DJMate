"""
chat_router.py — FastAPI router that exposes the SemanticInterpreter
to the React frontend via two endpoints:

  POST /chat/interpret   → parse a natural-language query into params
  POST /chat/search      → run the full search and return enriched tracks

Mount in your main FastAPI app:
  from chat_router import router as chat_router
  app.include_router(chat_router, prefix="/chat", tags=["chat"])
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging
import numpy as np
import json

from backend.tenant import get_tenant_db
from backend.data.db_interface import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class InterpretRequest(BaseModel):
    query: str
    current_track: Optional[Dict[str, Any]] = None  # currently playing track context


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
    album_art_url:    Optional[str]       = None
    semantic_tags:    Optional[List[str]] = []
    vibe_descriptors: Optional[List[str]] = []
    relevance_score:  Optional[float]     = None
    inferred:         Optional[bool]      = False


class TrackCandidate(BaseModel):
    """A fuzzy-matched track name result for find_similar_track intent."""
    trackid:     str
    title:       str
    artist:      str
    bpm:         Optional[float] = None
    key:         Optional[str]   = None
    match_score: float
    match_field: str


class SearchResponse(BaseModel):
    tracks:           List[TrackResult]
    relaxation_step:  int
    relaxation_label: str
    inferred_count:   int
    track_count:      int
    reasoning:        str
    confidence:       float
    model_used:       str
    intent:           str = "vibe_genre_search"
    # find_similar_track only
    track_candidates: Optional[List[TrackCandidate]] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_embedding(emb):
    if emb is None:
        return None
    if isinstance(emb, list):
        return [float(x) for x in emb]
    if isinstance(emb, str):
        try:
            parsed = json.loads(emb)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    return None


async def _embedding_search_for_track(db, track_id: str, limit: int) -> List[Dict[str, Any]]:
    """Run embedding similarity search for a given trackid. Returns plain dicts."""
    source = await db.get_track_by_id(track_id)
    if not source:
        return []

    raw_emb = source.embedding if hasattr(source, "embedding") else (
        source.get("embedding") if isinstance(source, dict) else None
    )
    source_emb = _parse_embedding(raw_emb)
    if not source_emb:
        return []

    raw = await db.find_similar_tracks(
        query_embedding=source_emb,
        limit=limit + 1,
        threshold=0.2,
    )

    results = []
    for r in raw:
        rid = str(r.get("id") or r.get("trackid") or "")
        if rid == str(track_id):
            continue

        full = await db.get_track_by_id(rid)
        if not full:
            continue

        sim = float(r.get("similarity", 0.5))
        target_raw = full.embedding if hasattr(full, "embedding") else (
            full.get("embedding") if isinstance(full, dict) else None
        )
        target_emb = _parse_embedding(target_raw)
        if target_emb and source_emb:
            a = np.array(source_emb, dtype=np.float32)
            b = np.array(target_emb, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na and nb:
                sim = float(np.dot(a, b) / (na * nb))

        results.append({
            "trackid":        str(full.trackid if hasattr(full, "trackid") else rid),
            "title":          full.title if hasattr(full, "title") else "Unknown",
            "artist":         full.artist if hasattr(full, "artist") else "Unknown",
            "bpm":            full.bpm if hasattr(full, "bpm") else None,
            "key":            full.key if hasattr(full, "key") else None,
            "energy":         full.energy if hasattr(full, "energy") else None,
            "semantic_tags":  full.semantic_tags if hasattr(full, "semantic_tags") else [],
            "vibe_descriptors": full.vibe_descriptors if hasattr(full, "vibe_descriptors") else [],
            "_relevance_score": round(sim, 4),
            "_inferred":      False,
        })
        if len(results) >= limit:
            break

    results.sort(key=lambda x: x["_relevance_score"], reverse=True)
    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/interpret", response_model=InterpretResponse)
async def interpret(req: InterpretRequest, db: DatabaseManager = Depends(get_tenant_db)):
    """
    Parse a free-text DJ query into structured search parameters.
    Now includes intent classification and current_track context.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from backend.llm_interpreter import SemanticInterpreter, InterpretationContext

    context = None
    if req.current_track:
        context = InterpretationContext(current_track=req.current_track)

    try:
        interp = SemanticInterpreter(supabase_client=db.client)
        await interp.initialize()
        params = await interp.interpret(req.query.strip(), context=context)
        return InterpretResponse(params=params)
    except Exception as e:
        logger.exception("Interpret error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, db: DatabaseManager = Depends(get_tenant_db)):
    """
    Run the full search pipeline.

    Routes based on intent:
    - find_similar_track: uses embedding similarity on the best candidate match
    - vibe_genre_search / transition_from_current: uses the tag-scoring pipeline
    """
    try:
        from backend.llm_interpreter import SemanticInterpreter
        interp = SemanticInterpreter(supabase_client=db.client)
        await interp.initialize()
        params = req.params
        intent = params.get("intent", "vibe_genre_search")

        # ── find_similar_track path ───────────────────────────────────────────
        if intent == "find_similar_track":
            candidates = params.get("track_candidates") or []
            track_count = max(1, int(params.get("track_count", 7)))

            if not candidates:
                # No candidates found — return empty with explanation
                return SearchResponse(
                    tracks=[],
                    relaxation_step=0,
                    relaxation_label="track not found",
                    inferred_count=0,
                    track_count=track_count,
                    reasoning=params.get("reasoning", "No matching track found in library."),
                    confidence=params.get("confidence", 0.0),
                    model_used=params.get("model_used", "intent-classifier"),
                    intent=intent,
                    track_candidates=[],
                )

            # Use the top candidate (highest match_score) for embedding search
            best = candidates[0]
            best_id = best.get("trackid", "")
            logger.info(
                f"🎵 find_similar_track → '{best.get('title')}' by {best.get('artist')} "
                f"(match_score={best.get('match_score', 0):.2f})"
            )

            # Fetch the matched track itself to put it first in results
            matched_track_full = await db.get_track_by_id(best_id)
            modifier = (params.get("modifier") or "").lower().strip()

            # Fetch more candidates than needed so we can re-rank with modifier
            fetch_count = track_count * 3 if modifier else track_count
            similar_raw = await _embedding_search_for_track(db, best_id, fetch_count)

            # Build the matched track as the first result (relevance_score=1.0)
            matched_result = None
            if matched_track_full:
                def _attr(obj, key, default=None):
                    return getattr(obj, key, None) if hasattr(obj, key) else (
                        obj.get(key, default) if isinstance(obj, dict) else default
                    )
                matched_result = TrackResult(
                    trackid=          str(_attr(matched_track_full, "trackid", best_id)),
                    title=            _attr(matched_track_full, "title", best.get("title", "Unknown")),
                    artist=           _attr(matched_track_full, "artist", best.get("artist", "Unknown")),
                    bpm=              _attr(matched_track_full, "bpm"),
                    key=              _attr(matched_track_full, "key"),
                    energy=           _attr(matched_track_full, "energy"),
                    filepath=         _attr(matched_track_full, "filepath"),
                    album_art_url=    _attr(matched_track_full, "album_art_url"),
                    semantic_tags=    _attr(matched_track_full, "semantic_tags") or [],
                    vibe_descriptors= _attr(matched_track_full, "vibe_descriptors") or [],
                    relevance_score=  1.0,
                    inferred=         False,
                )

            # Re-rank by modifier if present
            if modifier and similar_raw:
                source_energy = float(
                    (matched_track_full.energy if hasattr(matched_track_full, "energy") else None)
                    or (matched_track_full.get("energy") if isinstance(matched_track_full, dict) else None)
                    or 5
                )
                def _modifier_score(t: dict) -> float:
                    e = float(t.get("energy") or source_energy)
                    sim = t.get("_relevance_score", 0.5)
                    # Direction keywords → energy delta scoring
                    if any(k in modifier for k in ("higher energy", "more energy", "harder", "faster", "bigger")):
                        energy_score = (e - source_energy) / 10.0  # positive = higher energy
                    elif any(k in modifier for k in ("lower energy", "less energy", "softer", "slower", "calmer", "chilled")):
                        energy_score = (source_energy - e) / 10.0  # positive = lower energy
                    elif any(k in modifier for k in ("darker", "heavier", "deeper")):
                        energy_score = (e - source_energy) / 10.0
                    elif any(k in modifier for k in ("lighter", "brighter", "melodic")):
                        energy_score = (source_energy - e) / 10.0
                    else:
                        energy_score = 0.0
                    # Blend: 60% embedding similarity, 40% modifier direction
                    return sim * 0.6 + energy_score * 0.4

                similar_raw = sorted(similar_raw, key=_modifier_score, reverse=True)
                similar_raw = similar_raw[:track_count]
                logger.info(f"🎛️ Modifier '{modifier}' applied — re-ranked {len(similar_raw)} tracks")

            similar_tracks = [
                TrackResult(
                    trackid=          t["trackid"],
                    title=            t.get("title", "Unknown"),
                    artist=           t.get("artist", "Unknown"),
                    bpm=              t.get("bpm"),
                    key=              t.get("key"),
                    energy=           t.get("energy"),
                    album_art_url=    t.get("album_art_url"),
                    semantic_tags=    t.get("semantic_tags") or [],
                    vibe_descriptors= t.get("vibe_descriptors") or [],
                    relevance_score=  t.get("_relevance_score", 0.5),
                    inferred=         False,
                )
                for t in similar_raw
            ]

            tracks = ([matched_result] if matched_result else []) + similar_tracks

            return SearchResponse(
                tracks=tracks,
                relaxation_step=0,
                relaxation_label="exact match + similar",
                inferred_count=0,
                track_count=len(tracks),
                reasoning=(
                    f"Found '{best.get('title')}' by {best.get('artist')}. "
                    + (f"Re-ranked for '{modifier}'. " if modifier else "")
                    + f"Showing the track + {len(similar_tracks)} similar."
                ),
                confidence=best.get("match_score", 0.85),
                model_used=params.get("model_used", "embedding similarity"),
                intent=intent,
                track_candidates=[
                    TrackCandidate(
                        trackid=    c.get("trackid", ""),
                        title=      c.get("title", ""),
                        artist=     c.get("artist", ""),
                        bpm=        c.get("bpm"),
                        key=        c.get("key"),
                        match_score=c.get("match_score", 0.0),
                        match_field=c.get("match_field", ""),
                    )
                    for c in candidates
                ],
            )

        # ── vibe_genre_search / transition_from_current path ─────────────────
        tracks_raw, meta = await interp.search(params, db_manager=db)

        tracks = [
            TrackResult(
                trackid=          str(t.get("trackid") or t.get("id") or ""),
                title=            t.get("title",   "Unknown"),
                artist=           t.get("artist",  "Unknown"),
                bpm=              t.get("bpm"),
                key=              t.get("key"),
                energy=           t.get("energy"),
                filepath=         t.get("filepath"),
                album_art_url=    t.get("album_art_url"),
                semantic_tags=    t.get("semantic_tags")    or [],
                vibe_descriptors= t.get("vibe_descriptors") or t.get("vibe") or [],
                relevance_score=  t.get("_relevance_score", 0.5),
                inferred=         t.get("_inferred", False),
            )
            for t in tracks_raw
        ]

        return SearchResponse(
            tracks=           tracks,
            relaxation_step=  meta.get("relaxation_step",  0),
            relaxation_label= meta.get("relaxation_label", ""),
            inferred_count=   meta.get("inferred_count",   0),
            track_count=      params.get("track_count", len(tracks)),
            reasoning=        params.get("reasoning",  ""),
            confidence=       params.get("confidence", 0.0),
            model_used=       params.get("model_used", "fallback"),
            intent=           intent,
        )

    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health():
    """Quick liveness check for the chat endpoints."""
    return {"status": "ok", "endpoints": ["/chat/interpret", "/chat/search"]}