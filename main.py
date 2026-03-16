'''
Run uvicorn main:app --reload --host 0.0.0.0 --port 8000


'''

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
import json
import numpy as np

from backend.llm_interpreter import SemanticInterpreter
from backend.reccomender import DJRecommendationEngine
from backend.data.db_interface import DatabaseManager
from backend.chat_router import router as chat_router
from backend.tag_router import router as tag_router
from backend.crate_router import router as crate_router
from backend.playlist_router import router as playlist_router
app = FastAPI(title="AI DJ Curation API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(tag_router, prefix="/tags", tags=["tags"])
app.include_router(crate_router, prefix="/crates", tags=["crates"])
app.include_router(playlist_router, prefix="/playlists", tags=["playlists"])
# ── Request models ────────────────────────────────────────────────────────────

class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class StructuredQuery(BaseModel):
    tags: List[str] = []
    vibe_descriptors: List[str] = []
    bpm_range: Optional[tuple[int, int]] = None
    energy_range: Optional[tuple[float, float]] = None
    key_compatibility: Optional[str] = None
    direction: Optional[str] = None
    exclude_tracks: List[str] = []

class CrateOperation(BaseModel):
    session_id: str
    tracks: List[str]
    sequence_order: List[int]
    metadata: Optional[Dict[str, Any]] = None

# ── Singletons ────────────────────────────────────────────────────────────────

db_manager = DatabaseManager()
embedding_index = None
semantic_interpreter = SemanticInterpreter(
    supabase_client=db_manager.client
)
recommendation_engine = DJRecommendationEngine(
    db_interface=db_manager,
    embedding_index=None
)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "AI DJ Curation API",
        "version": "2.0.0"
    }

# ── Track library endpoints ───────────────────────────────────────────────────

@app.get("/tracks")
async def get_all_tracks(
        limit: int = 100,
        offset: int = 0
):
    """
    Get tracks with PRECOMPUTED UMAP coordinates for 3D visualization.

    Returns tracks with their x_coord, y_coord, z_coord from the database.
    These coordinates are precomputed by the 3d_Coordinator.py script using UMAP.

    Pagination prevents memory exhaustion on large libraries.

    Args:
        limit: Maximum tracks per request (default 100, max 500)
        offset: Starting position for pagination
    """
    try:
        if not db_manager.client:
            return {"tracks": [], "total": 0, "limit": limit, "offset": offset}

        # Limit maximum to prevent memory issues
        limit = min(limit, 500)

        # Get total count first (lightweight query)
        count_response = db_manager.client.table("tracks") \
            .select("trackid", count="exact") \
            .execute()
        total_count = count_response.count if hasattr(count_response, 'count') else \
            len(count_response.data or [])

        # Fetch paginated tracks WITH UMAP coordinates
        # CRITICAL: Include x_coord, y_coord, z_coord from your database
        response = db_manager.client.table("tracks") \
            .select("trackid, title, artist, album, bpm, key, filepath, "
                    "x_coord, y_coord, z_coord, "
                    "track_labels(semantic_tags, energy, vibe)") \
            .range(offset, offset + limit - 1) \
            .execute()

        tracks = []
        for track in response.data or []:
            labels = track.get("track_labels") or {}
            if isinstance(labels, list):
                labels = labels[0] if labels else {}

            # Get UMAP coordinates from database
            x_coord = track.get("x_coord")
            y_coord = track.get("y_coord")
            z_coord = track.get("z_coord")

            track_data = {
                "id":       track["trackid"],
                "trackid":  track["trackid"],
                "title":    track.get("title", "Unknown"),
                "artist":   track.get("artist", "Unknown"),
                "album":    track.get("album"),
                "bpm":      track.get("bpm"),
                "key":      track.get("key"),
                "filepath": track.get("filepath"),
                "energy":   labels.get("energy", 0.5),
                "tags":     labels.get("semantic_tags", []),
                "vibe":     labels.get("vibe", []),
            }

            # Add UMAP position if available
            if x_coord is not None and y_coord is not None and z_coord is not None:
                track_data["position"] = [
                    float(x_coord),
                    float(y_coord),
                    float(z_coord)
                ]
            else:
                # Fallback to None if no UMAP coordinates yet
                track_data["position"] = None

            tracks.append(track_data)

        return {
            "tracks": tracks,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tracks: {str(e)}")


@app.get("/tracks/resolve")
async def resolve_track(q: str):
    """
    Fuzzy-match a track name/artist string to a trackid.
    Used by the chatbox to resolve 'similar to Song X' queries.
    Must be defined BEFORE /tracks/{track_id} routes so FastAPI doesn't
    capture 'resolve' as a track_id.
    """
    try:
        if not db_manager.client or not q.strip():
            return {"match": None}

        search = q.strip().lower()

        # Search by title (ilike = case-insensitive)
        resp = db_manager.client.table("tracks") \
            .select("trackid, title, artist") \
            .ilike("title", f"%{search}%") \
            .limit(5) \
            .execute()

        if resp.data:
            return {"match": resp.data[0]}

        # Fallback: search by artist
        resp = db_manager.client.table("tracks") \
            .select("trackid, title, artist") \
            .ilike("artist", f"%{search}%") \
            .limit(5) \
            .execute()

        if resp.data:
            return {"match": resp.data[0]}

        return {"match": None}

    except Exception as e:
        return {"match": None}


@app.get("/tracks/{track_id}/neighbors")
async def get_track_neighbors(track_id: str, limit: int = 8):
    """
    Similarity edges for click-to-edges feature in the 3D cloud.
    Returns the top-N most similar tracks to use as edge targets.
    """
    try:
        track = await db_manager.get_track_by_id(track_id)
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")

        neighbors = []

        # Use embedding similarity if available
        embedding = track.embedding if hasattr(track, "embedding") else (
            track.get("embedding") if isinstance(track, dict) else None
        )

        if embedding:
            similar = await db_manager.find_similar_tracks(
                query_embedding=embedding,
                limit=limit + 1,
                threshold=0.3
            )
            for s in similar:
                sid = s.get("trackid") or s.get("id")
                if sid and sid != track_id:
                    neighbors.append({
                        "id": sid,
                        "similarity_score": float(s.get("similarity", 0.5)),
                    })
                if len(neighbors) >= limit:
                    break

        return {"source_id": track_id, "neighbors": neighbors}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch neighbors: {str(e)}")


@app.get("/tracks/{track_id}/audio")
async def get_track_audio(track_id: str):
    """Stream audio file for a given track. Used by the React frontend."""
    try:
        # Query Supabase directly to avoid stale cache in db_manager
        _direct = db_manager.client.table("tracks").select("filepath").eq("trackid", track_id).single().execute()
        filepath = (_direct.data or {}).get("filepath") if _direct.data else None

        if not filepath:
            raise HTTPException(status_code=404, detail="No filepath for this track")

        if not os.path.isfile(filepath):
            raise HTTPException(status_code=404, detail="Audio file not found on disk")

        ext = os.path.splitext(filepath)[1].lower()
        media_types = {
            '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
            '.aac': 'audio/aac', '.ogg': 'audio/ogg', '.m4a': 'audio/mp4',
            '.aiff': 'audio/aiff', '.aif': 'audio/aiff',
        }

        return FileResponse(
            path=filepath,
            media_type=media_types.get(ext, 'audio/mpeg'),
            filename=os.path.basename(filepath),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve audio: {str(e)}")


def _parse_embedding(emb):
    """Parse embedding that may be a string, list, or None. Returns list of floats or None."""
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


@app.get("/tracks/{track_id}/similar")
async def get_similar_tracks(track_id: str, limit: int = 7):
    """
    Embedding-based similarity search — same logic as streamSimilar.py Tab 1.
    Returns tracks ranked by cosine similarity of their embeddings.
    """
    try:
        # 1. Get source track (includes embedding via select *)
        source = await db_manager.get_track_by_id(track_id)
        if not source:
            raise HTTPException(status_code=404, detail="Track not found")

        raw_emb = source.embedding if hasattr(source, 'embedding') else (
            source.get('embedding') if isinstance(source, dict) else None
        )
        source_embedding = _parse_embedding(raw_emb)
        if not source_embedding:
            raise HTTPException(status_code=404, detail="Track has no embedding")

        # 2. Find similar via Supabase RPC (match_tracks) — same as streamSimilar.py
        raw = await db_manager.find_similar_tracks(
            query_embedding=source_embedding,
            limit=limit + 1,
            threshold=0.2
        )

        # 3. Filter out source, enrich with full data, compute cosine similarity
        similar = []
        for r in raw:
            rid = str(r.get("id") or r.get("trackid") or "")
            if rid == str(track_id):
                continue

            # Get full track data
            full = await db_manager.get_track_by_id(rid)
            if not full:
                continue

            # Cosine similarity (same as streamSimilar.py)
            sim = float(r.get("similarity", 0.5))
            target_raw = full.embedding if hasattr(full, 'embedding') else (
                full.get('embedding') if isinstance(full, dict) else None
            )
            target_emb = _parse_embedding(target_raw)
            if target_emb and source_embedding:
                a = np.array(source_embedding, dtype=np.float32)
                b = np.array(target_emb, dtype=np.float32)
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na and nb:
                    sim = float(np.dot(a, b) / (na * nb))

            # Extract labels
            tags = []
            energy = 0.5
            vibes = []
            if hasattr(full, 'semantic_tags') and full.semantic_tags:
                tags = full.semantic_tags
            if hasattr(full, 'energy') and full.energy is not None:
                energy = full.energy
            if hasattr(full, 'vibe_descriptors') and full.vibe_descriptors:
                vibes = full.vibe_descriptors

            similar.append({
                "trackid": str(full.trackid if hasattr(full, 'trackid') else rid),
                "title": full.title if hasattr(full, 'title') else "Unknown",
                "artist": full.artist if hasattr(full, 'artist') else "Unknown",
                "bpm": full.bpm if hasattr(full, 'bpm') else None,
                "key": full.key if hasattr(full, 'key') else None,
                "energy": energy,
                "semantic_tags": tags,
                "vibe_descriptors": vibes,
                "relevance_score": round(sim, 4),
            })

            if len(similar) >= limit:
                break

        # Sort by similarity descending
        similar.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Source track info for direction badges
        source_info = {
            "trackid": str(source.trackid if hasattr(source, 'trackid') else track_id),
            "title": source.title if hasattr(source, 'title') else "Unknown",
            "artist": source.artist if hasattr(source, 'artist') else "Unknown",
            "bpm": source.bpm if hasattr(source, 'bpm') else None,
            "key": source.key if hasattr(source, 'key') else None,
            "energy": source.energy if hasattr(source, 'energy') else 0.5,
            "semantic_tags": source.semantic_tags if hasattr(source, 'semantic_tags') else [],
            "vibe_descriptors": source.vibe_descriptors if hasattr(source, 'vibe_descriptors') else [],
        }

        return {
            "source": source_info,
            "tracks": similar,
            "track_count": len(similar),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


# ── AI endpoints ──────────────────────────────────────────────────────────────

@app.post("/parse-intent")
async def parse_natural_language(query: NaturalLanguageQuery):
    """Convert natural language to structured query parameters."""
    try:
        context = await db_manager.get_session_context(query.session_id) \
            if query.session_id else None

        structured_query = await semantic_interpreter.interpret(
            query.query,
            context=context
        )

        return {
            "original_query":       query.query,
            "structured_query":     structured_query,
            "confidence":           structured_query.get("confidence", 0.0),
            "interpretation_notes": structured_query.get("notes", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent parsing failed: {str(e)}")


@app.post("/intelligent-recommend")
async def intelligent_recommend(
        query: StructuredQuery,
        context_track_id: Optional[str] = None
):
    """Get recommendations using structured parameters with DJ logic."""
    try:
        recommendations = await recommendation_engine.get_intelligent_recommendations(
            structured_query=query,
            context_track_id=context_track_id,
            apply_harmonic_weighting=True,
            apply_energy_flow=True,
            diversity_penalty=0.2
        )

        return {
            "recommendations":      recommendations,
            "reasoning":            recommendations.get("reasoning", {}),
            "compatibility_scores": recommendations.get("compatibility", {}),
            "pathway_visualization":recommendations.get("pathway_data", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/crate/operations")
async def manage_crate(operation: CrateOperation):
    """Handle crate operations with compatibility validation."""
    try:
        compatibility_issues = await recommendation_engine.validate_sequence(
            operation.tracks,
            operation.sequence_order
        )

        await db_manager.update_crate(
            session_id=operation.session_id,
            tracks=operation.tracks,
            sequence=operation.sequence_order,
            validation_results=compatibility_issues
        )

        return {
            "success":              True,
            "compatibility_issues": compatibility_issues,
            "sequence_score":       compatibility_issues.get("overall_score", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crate operation failed: {str(e)}")


@app.get("/visualization/pathway")
async def get_pathway_visualization(from_track: str, to_tracks: List[str]):
    """Generate 3D pathway data for visualization."""
    try:
        pathway_data = await recommendation_engine.generate_pathway_visualization(
            source_track=from_track,
            target_tracks=to_tracks,
            max_intermediate_steps=3
        )
        return pathway_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pathway generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)