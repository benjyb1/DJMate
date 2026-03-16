"""
tag_router.py — FastAPI router for track label CRUD, correction logging, and online learning.

Endpoints:
  GET  /tags/available           — all unique tags and vibes (for autocomplete)
  GET  /tags/{trackid}           — current labels for a track
  PUT  /tags/{trackid}           — update labels (sets tag_source='manual', logs corrections)
  POST /tags/learning/propagate  — trigger online learning propagation
  GET  /tags/learning/status     — correction count, patterns, last propagation

Mount in your main FastAPI app:
  from backend.tag_router import router as tag_router
  app.include_router(tag_router, prefix="/tags", tags=["tags"])
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy singletons (initialised on first request) ───────────────────────────
_db = None
_learner = None


def _get_db():
    global _db
    if _db is None:
        from backend.data.db_interface import DatabaseManager
        _db = DatabaseManager(enable_caching=True, pool_size=10)
    return _db


def _get_learner():
    global _learner
    if _learner is None:
        from backend.online_learner import OnlineLearner
        _learner = OnlineLearner()
    return _learner


# ── Request / Response models ────────────────────────────────────────────────

class UpdateLabelsRequest(BaseModel):
    semantic_tags: Optional[List[str]] = None
    vibe: Optional[List[str]] = None
    energy: Optional[float] = Field(None, ge=0.0, le=1.0)


class TrackLabels(BaseModel):
    trackid: str
    semantic_tags: List[str] = []
    vibe: List[str] = []
    energy: Optional[float] = None
    tag_source: Optional[str] = None


class AvailableLabels(BaseModel):
    semantic_tags: List[str] = []
    vibes: List[str] = []


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/available", response_model=AvailableLabels)
async def get_available_tags():
    """Return all unique semantic_tags and vibes across the library (for autocomplete)."""
    try:
        db = _get_db()
        result = await db.get_available_tags()
        return AvailableLabels(
            semantic_tags=result.get("semantic_tags", []),
            vibes=result.get("vibes", []),
        )
    except Exception as e:
        logger.error(f"get_available_tags error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{trackid}", response_model=TrackLabels)
async def get_track_labels(trackid: str):
    """Get current labels for a single track."""
    try:
        db = _get_db()
        labels = await db.get_track_labels(trackid)
        if labels is None:
            raise HTTPException(status_code=404, detail="Track labels not found")
        return TrackLabels(
            trackid=labels["trackid"],
            semantic_tags=db._parse_json_field(labels.get("semantic_tags")),
            vibe=db._parse_json_field(labels.get("vibe")),
            energy=labels.get("energy"),
            tag_source=labels.get("tag_source"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_track_labels error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{trackid}", response_model=TrackLabels)
async def update_track_labels(trackid: str, req: UpdateLabelsRequest):
    """
    Update labels for a track.

    Sets tag_source='manual' and logs old->new changes to tag_corrections.
    Only the fields present in the request body are updated.
    """
    updates = {}
    if req.semantic_tags is not None:
        updates["semantic_tags"] = req.semantic_tags
    if req.vibe is not None:
        updates["vibe"] = req.vibe
    if req.energy is not None:
        updates["energy"] = req.energy

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        db = _get_db()
        result = await db.update_track_labels(trackid, updates, log_corrections=True)
        return TrackLabels(
            trackid=result["trackid"],
            semantic_tags=db._parse_json_field(result.get("semantic_tags")),
            vibe=db._parse_json_field(result.get("vibe")),
            energy=result.get("energy"),
            tag_source=result.get("tag_source"),
        )
    except Exception as e:
        logger.error(f"update_track_labels error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Learning endpoints ───────────────────────────────────────────────────────

@router.post("/learning/propagate")
async def trigger_propagation(dry_run: bool = False) -> Dict[str, Any]:
    """
    Trigger online learning propagation.

    Finds correction patterns from recent user edits and applies confident
    patterns to auto-tagged tracks. Pass ?dry_run=true to preview without
    writing changes.
    """
    try:
        learner = _get_learner()
        result = await learner.propagate(dry_run=dry_run)
        return result
    except Exception as e:
        logger.error(f"propagation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/status")
async def get_learning_status() -> Dict[str, Any]:
    """Return correction count, pattern count, and last propagation time."""
    try:
        learner = _get_learner()
        return await learner.get_status()
    except Exception as e:
        logger.error(f"learning status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
