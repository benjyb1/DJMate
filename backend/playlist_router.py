"""
playlist_router.py — FastAPI router for playlist CRUD and Rekordbox import/export.

Endpoints:
  POST /playlists/import       — upload Rekordbox XML, parse, match, create playlist
  GET  /playlists               — list all playlists
  GET  /playlists/{id}          — get playlist with tracks
  PUT  /playlists/{id}/tracks   — reorder/add/remove tracks
  POST /playlists/{id}/suggest  — LLM suggests tracks that fit the playlist
  GET  /playlists/{id}/export   — export as Rekordbox XML

Mount: app.include_router(router, prefix="/playlists", tags=["playlists"])
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import uuid
from xml.etree.ElementTree import ParseError as ET_ParseError

from backend.rekordbox_parser import parse_xml, match_tracks, generate_export_xml

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Lazy singleton (initialised on first request) ────────────────────────────

_db = None


def _get_db():
    global _db
    if _db is None:
        from backend.data.db_interface import DatabaseManager
        _db = DatabaseManager(enable_caching=True, pool_size=10)
    return _db


# ── Request / Response models ────────────────────────────────────────────────

class ImportRequest(BaseModel):
    xml_content: str
    playlist_name: Optional[str] = None  # override name from XML


class ImportResult(BaseModel):
    playlist_id: str
    playlist_name: str
    total_xml_tracks: int
    matched_tracks: int
    unmatched_tracks: int
    matches: List[Dict[str, Any]] = []


class PlaylistSummary(BaseModel):
    id: str
    name: str
    source: Optional[str] = None
    track_count: int = 0
    created_at: Optional[str] = None


class PlaylistDetail(BaseModel):
    id: str
    name: str
    source: Optional[str] = None
    tracks: List[Dict[str, Any]] = []


class UpdateTracksRequest(BaseModel):
    track_ids: List[str]  # ordered list of track IDs


class SuggestRequest(BaseModel):
    prompt: Optional[str] = None  # optional natural-language hint
    count: int = 5


class SuggestResponse(BaseModel):
    suggestions: List[Dict[str, Any]] = []


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/import", response_model=ImportResult)
async def import_rekordbox(req: ImportRequest):
    """Parse Rekordbox XML, fuzzy-match tracks to DB, and create a playlist."""
    try:
        db = _get_db()

        # Parse the XML
        parsed = parse_xml(req.xml_content)
        xml_tracks = parsed["collection"]
        xml_playlists = parsed["playlists"]

        # Choose playlist name: explicit override > first XML playlist > default
        name = req.playlist_name
        if not name and xml_playlists:
            name = xml_playlists[0]["name"]
        if not name:
            name = "Imported Playlist"

        # Get all DB tracks for matching
        all_db_tracks = await _fetch_all_db_tracks(db)

        # Determine which XML tracks belong to the target playlist
        target_keys = set()
        if xml_playlists:
            target_keys = set(xml_playlists[0]["track_keys"])

        # Filter collection to playlist tracks (or use all if no playlist structure)
        if target_keys:
            playlist_xml_tracks = [t for t in xml_tracks if t["trackid"] in target_keys]
        else:
            playlist_xml_tracks = xml_tracks

        # Fuzzy match
        matches = match_tracks(playlist_xml_tracks, all_db_tracks)

        # Create playlist with matched track IDs
        matched_ids = [m["db_trackid"] for m in matches if m["db_trackid"] is not None]
        playlist_id = str(uuid.uuid4())
        await db.create_playlist(playlist_id, name, source="rekordbox")

        if matched_ids:
            await db.update_playlist_tracks(playlist_id, matched_ids)

        matched_count = len(matched_ids)
        return ImportResult(
            playlist_id=playlist_id,
            playlist_name=name,
            total_xml_tracks=len(playlist_xml_tracks),
            matched_tracks=matched_count,
            unmatched_tracks=len(playlist_xml_tracks) - matched_count,
            matches=[
                {
                    "xml_name": m["xml_track"].get("name", ""),
                    "xml_artist": m["xml_track"].get("artist", ""),
                    "db_trackid": m["db_trackid"],
                    "confidence": m["confidence"],
                }
                for m in matches
            ],
        )

    except ET_ParseError as e:
        raise HTTPException(status_code=400, detail=f"Invalid XML: {e}")
    except Exception as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[PlaylistSummary])
async def list_playlists():
    """List all playlists."""
    try:
        db = _get_db()
        playlists = await db.get_playlists()
        return [
            PlaylistSummary(
                id=p["id"],
                name=p["name"],
                source=p.get("source"),
                track_count=p.get("track_count", 0),
                created_at=p.get("created_at"),
            )
            for p in playlists
        ]
    except Exception as e:
        logger.error(f"List playlists error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{playlist_id}", response_model=PlaylistDetail)
async def get_playlist(playlist_id: str):
    """Get a playlist with its tracks."""
    try:
        db = _get_db()
        playlist = await db.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")
        return PlaylistDetail(**playlist)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{playlist_id}/tracks", response_model=PlaylistDetail)
async def update_playlist_tracks(playlist_id: str, req: UpdateTracksRequest):
    """Reorder, add, or remove tracks in a playlist."""
    try:
        db = _get_db()

        # Verify playlist exists
        playlist = await db.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        await db.update_playlist_tracks(playlist_id, req.track_ids)

        # Return updated playlist
        updated = await db.get_playlist(playlist_id)
        return PlaylistDetail(**updated)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update playlist tracks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{playlist_id}/suggest", response_model=SuggestResponse)
async def suggest_tracks(playlist_id: str, req: SuggestRequest):
    """Use the LLM to suggest tracks that fit the playlist's vibe."""
    try:
        db = _get_db()

        playlist = await db.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        existing_tracks = playlist.get("tracks", [])
        existing_ids = {t.get("trackid") for t in existing_tracks}

        # Build a prompt from the playlist context
        prompt = req.prompt or _build_suggest_prompt(existing_tracks)

        # Use the LLM interpreter to find matching tracks
        from backend.llm_interpreter import SemanticInterpreter, InterpretationContext
        interp = SemanticInterpreter(supabase_client=db.client)
        await interp.initialize()

        context = None
        if existing_tracks:
            context = InterpretationContext(current_track=existing_tracks[-1])

        params = await interp.interpret(prompt, context=context)

        # Query DB with interpreted params
        structured_query = {
            "semantic_tags": [t[0] for t in params.get("tag_scores", [])],
            "vibes": [v[0] for v in params.get("vibe_scores", [])],
            "bpm_range": params.get("bpm_range"),
            "energy_range": params.get("energy_range"),
        }

        candidates = await db.get_tracks_by_semantic_filter(
            structured_query, limit=req.count + len(existing_ids)
        )

        # Filter out tracks already in the playlist
        suggestions = []
        for track in candidates:
            tid = track.trackid if hasattr(track, "trackid") else track.get("trackid")
            if tid not in existing_ids:
                suggestions.append({
                    "trackid": tid,
                    "title": track.title if hasattr(track, "title") else track.get("title"),
                    "artist": track.artist if hasattr(track, "artist") else track.get("artist"),
                    "bpm": track.bpm if hasattr(track, "bpm") else track.get("bpm"),
                    "key": track.key if hasattr(track, "key") else track.get("key"),
                    "energy": track.energy if hasattr(track, "energy") else track.get("energy"),
                    "semantic_tags": track.semantic_tags if hasattr(track, "semantic_tags") else [],
                })
                if len(suggestions) >= req.count:
                    break

        return SuggestResponse(suggestions=suggestions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Suggest tracks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{playlist_id}/export")
async def export_playlist(playlist_id: str):
    """Export a playlist as Rekordbox XML."""
    try:
        db = _get_db()

        playlist = await db.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        xml_content = generate_export_xml(
            playlist_name=playlist["name"],
            tracks=playlist.get("tracks", []),
        )

        return Response(
            content=xml_content,
            media_type="application/xml",
            headers={
                "Content-Disposition": f'attachment; filename="{playlist["name"]}.xml"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{playlist_id}")
async def delete_playlist(playlist_id: str):
    """Delete a playlist and its track associations."""
    try:
        db = _get_db()
        success = await db.delete_playlist(playlist_id)
        if not success:
            raise HTTPException(status_code=404, detail="Playlist not found or delete failed")
        return {"status": "deleted", "id": playlist_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _fetch_all_db_tracks(db) -> List[dict]:
    """Fetch all tracks from DB as simple dicts for matching."""
    if not db.client:
        return []
    try:
        response = db.client.table("tracks") \
            .select("trackid, title, artist") \
            .execute()
        return response.data or []
    except Exception as e:
        logger.warning(f"Failed to fetch tracks for matching: {e}")
        return []


def _build_suggest_prompt(tracks: List[dict]) -> str:
    """Build a natural-language prompt from existing playlist tracks."""
    if not tracks:
        return "suggest some tracks"

    # Collect tags, vibes, avg BPM from existing tracks
    tags = set()
    vibes = set()
    bpms = []
    for t in tracks:
        for tag in (t.get("semantic_tags") or []):
            tags.add(tag)
        for vibe in (t.get("vibe_descriptors") or t.get("vibe") or []):
            vibes.add(vibe)
        if t.get("bpm"):
            bpms.append(float(t["bpm"]))

    parts = ["find tracks similar to this playlist"]
    if tags:
        parts.append(f"genres: {', '.join(list(tags)[:5])}")
    if vibes:
        parts.append(f"vibe: {', '.join(list(vibes)[:5])}")
    if bpms:
        avg_bpm = sum(bpms) / len(bpms)
        parts.append(f"around {avg_bpm:.0f} BPM")

    return ", ".join(parts)
