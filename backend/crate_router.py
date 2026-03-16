"""
crate_router.py — FastAPI router for crate-based set building.

Endpoints:
  POST /crates/session           — create a new crate session
  GET  /crates/session/{id}      — get full crate tree for a session
  POST /crates/generate          — LLM-powered crate generation
  POST /crates/{id}/tracks       — add/remove/reorder tracks in a crate
  DELETE /crates/{id}            — delete a crate and its children
  POST /crates/bridge            — analyze gap between two crates, suggest bridges
  POST /crates/reorder           — reorder crates within a session

Mount:
  from backend.crate_router import router as crate_router
  app.include_router(crate_router, prefix="/crates", tags=["crates"])
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

_db = None
_generator = None
_bridge_analyzer = None

def _get_db():
    global _db
    if _db is None:
        from backend.data.db_interface import DatabaseManager
        _db = DatabaseManager(enable_caching=True, pool_size=10)
    return _db

def _get_generator():
    global _generator
    if _generator is None:
        from backend.crate_generator import CrateGenerator
        _generator = CrateGenerator(_get_db())
    return _generator

def _get_bridge_analyzer():
    global _bridge_analyzer
    if _bridge_analyzer is None:
        from backend.crate_generator import BridgeAnalyzer
        _bridge_analyzer = BridgeAnalyzer(_get_db())
    return _bridge_analyzer

# ── Request/Response models ──

class CreateSessionRequest(BaseModel):
    name: Optional[str] = "Untitled Session"

class SessionResponse(BaseModel):
    id: str
    name: str
    crates: List[Dict[str, Any]] = []

class GenerateCrateRequest(BaseModel):
    session_id: str
    prompt: str
    parent_crate_id: Optional[str] = None
    current_track: Optional[Dict[str, Any]] = None  # for transition context

class CrateResponse(BaseModel):
    id: str
    session_id: str
    parent_crate_id: Optional[str] = None
    label: str
    tracks: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class BridgeRequest(BaseModel):
    crate_a_id: str
    crate_b_id: str

class BridgeResponse(BaseModel):
    gap_score: float
    bpm_gap: float
    energy_gap: float
    key_compatible: bool
    needs_bridge: bool
    bridge_tracks: List[Dict[str, Any]] = []

class UpdateTracksRequest(BaseModel):
    track_ids: List[str]  # ordered list of track IDs for this crate

# ── Endpoints ──

@router.post("/session", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest):
    """Create a new crate-building session."""
    try:
        db = _get_db()
        session_id = str(uuid.uuid4())
        # Store session in DB (if table exists) or in-memory
        session = await db.create_crate_session(session_id, req.name)
        return SessionResponse(id=session_id, name=req.name, crates=[])
    except Exception as e:
        logger.error(f"Create session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get full crate tree for a session."""
    try:
        db = _get_db()
        session = await db.get_crate_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse(**session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=CrateResponse)
async def generate_crate(req: GenerateCrateRequest):
    """
    Generate a crate using the LLM to interpret the prompt,
    then the crate generator to assemble cohesive tracks.
    """
    try:
        db = _get_db()
        generator = _get_generator()

        # Step 1: Interpret the prompt via the LLM
        from backend.llm_interpreter import SemanticInterpreter, InterpretationContext
        interp = SemanticInterpreter(supabase_client=db.client)
        await interp.initialize()

        context = None
        if req.current_track:
            context = InterpretationContext(current_track=req.current_track)

        params = await interp.interpret(req.prompt, context=context)

        # Step 2: Build CrateSpec from params
        from backend.crate_generator import CrateSpec
        spec = CrateSpec(
            session_id=req.session_id,
            parent_crate_id=req.parent_crate_id,
            genres=[{"name": t[0], "confidence": t[1]} for t in params.get("tag_scores", [])],
            vibes=[{"name": v[0], "confidence": v[1]} for v in params.get("vibe_scores", [])],
            energy_range=params.get("energy_range"),
            bpm_range=params.get("bpm_range"),
            track_count=params.get("track_count", 5),
            direction=params.get("crate_direction"),
            direction_desc=params.get("crate_direction_desc"),
        )

        # Step 3: Get parent crate's tracks if branching
        parent_tracks = []
        if req.parent_crate_id:
            parent_crate = await db.get_crate(req.parent_crate_id)
            if parent_crate:
                parent_tracks = parent_crate.get("tracks", [])

        # Step 4: Generate the crate
        result = await generator.generate_crate(spec, parent_tracks)

        # Step 5: Save to DB
        crate_id = str(uuid.uuid4())
        crate_data = {
            "id": crate_id,
            "session_id": req.session_id,
            "parent_crate_id": req.parent_crate_id,
            "label": result["label"],
            "tracks": result["tracks"],
            "metadata": result["metadata"],
        }
        await db.save_crate(crate_data)

        # Step 6: Auto-run bridge analysis if parent exists
        if req.parent_crate_id and parent_tracks and result["tracks"]:
            bridge_analyzer = _get_bridge_analyzer()
            bridge = await bridge_analyzer.analyze_bridge(parent_tracks, result["tracks"])
            if bridge.get("needs_bridge"):
                crate_data["metadata"]["bridge_warning"] = bridge

        return CrateResponse(**crate_data)

    except Exception as e:
        logger.error(f"Generate crate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bridge", response_model=BridgeResponse)
async def analyze_bridge(req: BridgeRequest):
    """Analyze the transition gap between two crates."""
    try:
        db = _get_db()
        bridge_analyzer = _get_bridge_analyzer()

        crate_a = await db.get_crate(req.crate_a_id)
        crate_b = await db.get_crate(req.crate_b_id)

        if not crate_a or not crate_b:
            raise HTTPException(status_code=404, detail="Crate not found")

        result = await bridge_analyzer.analyze_bridge(
            crate_a.get("tracks", []),
            crate_b.get("tracks", [])
        )
        return BridgeResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bridge analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{crate_id}/tracks")
async def update_crate_tracks(crate_id: str, req: UpdateTracksRequest):
    """Update the track list for a crate."""
    try:
        db = _get_db()
        await db.update_crate_tracks(crate_id, req.track_ids)
        crate = await db.get_crate(crate_id)
        return crate or {"id": crate_id, "tracks": req.track_ids}
    except Exception as e:
        logger.error(f"Update crate tracks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{crate_id}")
async def delete_crate(crate_id: str):
    """Delete a crate and all its children."""
    try:
        db = _get_db()
        await db.delete_crate(crate_id)
        return {"status": "deleted", "id": crate_id}
    except Exception as e:
        logger.error(f"Delete crate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
