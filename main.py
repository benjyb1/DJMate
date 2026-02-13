from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio

from Backend.llm_interpreter import SemanticInterpreter
from Backend.recommender import DJRecommendationEngine
from Backend.data.db_interface import DatabaseManager

app = FastAPI(title="AI DJ Curation API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
semantic_interpreter = SemanticInterpreter()
recommendation_engine = DJRecommendationEngine(
    db_manager=db_manager,
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