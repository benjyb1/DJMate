'''
Run uvicorn main:app --reload --host 0.0.0.0 --port 8000


'''

from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import logging
import numpy as np

from backend.llm_interpreter import SemanticInterpreter

from backend.dependencies import get_db
from backend.tenant import get_tenant_db
from backend.data.db_interface import DatabaseManager
from backend.chat_router import router as chat_router
from backend.tag_router import router as tag_router
from backend.crate_router import router as crate_router
from backend.playlist_router import router as playlist_router
from backend.ingest_router import router as ingest_router
from backend.suggested_playlist_router import router as suggested_playlist_router
from backend.setup_router import router as setup_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # DB init happens here (after uvicorn binds the port) rather than at
    # module-import time.  This lets Render's port scanner detect :10000
    # immediately instead of waiting for Supabase + Redis round-trips.
    logger.info("DJMate API starting up...")
    db_manager = get_db()
    try:
        test = db_manager.client.table("tracks").select("trackid").limit(1).execute()
        logger.info("Database connection OK")
    except Exception as e:
        logger.error(f"Database startup check failed: {e}")
    yield
    logger.info("DJMate API shutting down...")
    if hasattr(db_manager, 'pg_pool') and db_manager.pg_pool:
        await db_manager.pg_pool.close()
    if hasattr(db_manager, 'redis_client') and db_manager.redis_client:
        try:
            db_manager.redis_client.close()
        except Exception:
            pass


app = FastAPI(title="AI DJ Curation API", version="2.0.0", lifespan=lifespan)

# Configure CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Supabase-Url", "X-Supabase-Key"],
)
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(tag_router, prefix="/tags", tags=["tags"])
app.include_router(crate_router, prefix="/crates", tags=["crates"])
app.include_router(playlist_router, prefix="/playlists", tags=["playlists"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(suggested_playlist_router, prefix="/suggested-playlists", tags=["suggested-playlists"])
app.include_router(setup_router, prefix="/setup", tags=["setup"])
# ── Request models ────────────────────────────────────────────────────────────

class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


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
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        db: DatabaseManager = Depends(get_tenant_db),
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
        if not db.client:
            return {"tracks": [], "total": 0, "limit": limit, "offset": offset}

        # Limit maximum to prevent memory issues
        limit = min(limit, 500)

        # Get total count first (lightweight query)
        count_response = db.client.table("tracks") \
            .select("trackid", count="exact") \
            .execute()
        total_count = count_response.count if hasattr(count_response, 'count') else \
            len(count_response.data or [])

        # Fetch paginated tracks WITH UMAP coordinates
        # CRITICAL: Include x_coord, y_coord, z_coord from your database
        response = db.client.table("tracks") \
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
        logger.exception("Failed to fetch tracks")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/tracks/resolve")
async def resolve_track(q: str, db: DatabaseManager = Depends(get_tenant_db)):
    """
    Fuzzy-match a track name/artist string to a trackid.
    Used by the chatbox to resolve 'similar to Song X' queries.
    Must be defined BEFORE /tracks/{track_id} routes so FastAPI doesn't
    capture 'resolve' as a track_id.
    """
    try:
        if not db.client or not q.strip():
            return {"match": None}

        search = q.strip().lower()

        # Search by title (ilike = case-insensitive)
        resp = db.client.table("tracks") \
            .select("trackid, title, artist") \
            .ilike("title", f"%{search}%") \
            .limit(5) \
            .execute()

        if resp.data:
            return {"match": resp.data[0]}

        # Fallback: search by artist
        resp = db.client.table("tracks") \
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
async def get_track_neighbors(track_id: str, limit: int = Query(8, ge=1, le=100),
                              db: DatabaseManager = Depends(get_tenant_db)):
    """
    Similarity edges for click-to-edges feature in the 3D cloud.
    Returns the top-N most similar tracks to use as edge targets.
    """
    try:
        track = await db.get_track_by_id(track_id)
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")

        neighbors = []

        # Use embedding similarity if available
        embedding = track.embedding if hasattr(track, "embedding") else (
            track.get("embedding") if isinstance(track, dict) else None
        )

        if embedding:
            similar = await db.find_similar_tracks(
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
        logger.exception("Failed to fetch neighbors")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/tracks/{track_id}/audio")
async def get_track_audio(track_id: str, db: DatabaseManager = Depends(get_tenant_db)):
    """Stream audio file for a given track. Used by the React frontend."""
    try:
        # Query Supabase directly to avoid stale cache
        _direct = db.client.table("tracks").select("filepath").eq("trackid", track_id).single().execute()
        filepath = (_direct.data or {}).get("filepath") if _direct.data else None

        if not filepath:
            raise HTTPException(status_code=404, detail="No filepath for this track")

        # Path traversal protection
        VALID_AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg', '.aiff', '.aif', '.wma'}
        resolved = Path(filepath).resolve()
        if resolved.suffix.lower() not in VALID_AUDIO_EXTS:
            raise HTTPException(status_code=400, detail="Invalid audio file type")

        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")

        media_types = {
            '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
            '.aac': 'audio/aac', '.ogg': 'audio/ogg', '.m4a': 'audio/mp4',
            '.aiff': 'audio/aiff', '.aif': 'audio/aiff', '.wma': 'audio/x-ms-wma',
        }

        return FileResponse(
            path=str(resolved),
            media_type=media_types.get(resolved.suffix.lower(), 'audio/mpeg'),
            filename=resolved.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to serve audio")
        raise HTTPException(status_code=500, detail="Internal server error")


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
async def get_similar_tracks(track_id: str, limit: int = Query(7, ge=1, le=100),
                             db: DatabaseManager = Depends(get_tenant_db)):
    """
    Embedding-based similarity search — same logic as streamSimilar.py Tab 1.
    Returns tracks ranked by cosine similarity of their embeddings.
    """
    try:
        # 1. Get source track (includes embedding via select *)
        source = await db.get_track_by_id(track_id)
        if not source:
            raise HTTPException(status_code=404, detail="Track not found")

        raw_emb = source.embedding if hasattr(source, 'embedding') else (
            source.get('embedding') if isinstance(source, dict) else None
        )
        source_embedding = _parse_embedding(raw_emb)
        if not source_embedding:
            raise HTTPException(status_code=404, detail="Track has no embedding")

        # 2. Find similar via Supabase RPC (match_tracks) — same as streamSimilar.py
        raw = await db.find_similar_tracks(
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
            full = await db.get_track_by_id(rid)
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
        logger.exception("Similar search failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ── AI endpoints ──────────────────────────────────────────────────────────────

@app.post("/parse-intent")
async def parse_natural_language(query: NaturalLanguageQuery,
                                 db: DatabaseManager = Depends(get_tenant_db)):
    """Convert natural language to structured query parameters."""
    try:
        context = await db.get_session_context(query.session_id) \
            if query.session_id else None

        interp = SemanticInterpreter(supabase_client=db.client)
        await interp.initialize()
        structured_query = await interp.interpret(
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
        logger.exception("Intent parsing failed")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)