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
import os
import shutil
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


class CreatePlaylistRequest(BaseModel):
    name: str
    parent_id: Optional[str] = None
    description: Optional[str] = None


class RenamePlaylistRequest(BaseModel):
    name: str


class AddTracksRequest(BaseModel):
    track_ids: List[str]


class RemoveTracksRequest(BaseModel):
    track_ids: List[str]


class ExportFoldersRequest(BaseModel):
    playlist_ids: List[str]


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


@router.post("/create")
async def create_playlist(req: CreatePlaylistRequest):
    """Create an empty playlist with optional parent_id and description."""
    try:
        db = _get_db()
        playlist_id = str(uuid.uuid4())
        payload = {"id": playlist_id, "name": req.name, "source": "manual"}
        if req.parent_id:
            payload["parent_id"] = req.parent_id
        if req.description:
            payload["description"] = req.description

        # Insert directly since create_playlist() doesn't support parent_id/description
        if db.client:
            db.client.table("playlists").insert(payload).execute()

        return {"id": playlist_id, "name": req.name, "parent_id": req.parent_id, "description": req.description}
    except Exception as e:
        logger.error(f"Create playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tree")
async def get_playlist_tree():
    """Get all playlists as a flat list with track counts (for tree rendering)."""
    try:
        db = _get_db()
        tree = await db.get_playlist_tree()
        return tree
    except Exception as e:
        logger.error(f"Get playlist tree error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool")
async def get_track_pool():
    """Get all tracks with labels (lightweight, no embeddings)."""
    try:
        db = _get_db()
        pool = await db.get_track_pool()
        return pool
    except Exception as e:
        logger.error(f"Get track pool error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-folders")
async def export_folders(req: ExportFoldersRequest):
    """Export playlists as folders with copied audio files."""
    try:
        db = _get_db()
        all_filepaths = []
        playlist_data = []

        # Gather playlist names and track filepaths
        for pid in req.playlist_ids:
            playlist = await db.get_playlist(pid)
            if not playlist:
                logger.warning(f"Playlist {pid} not found, skipping")
                continue

            tracks = playlist.get("tracks", [])
            # Need filepaths — fetch from track pool or direct query
            track_ids = [t.get("trackid") for t in tracks]
            filepaths = []
            if db.client and track_ids:
                try:
                    resp = db.client.table("tracks") \
                        .select("trackid, filepath") \
                        .in_("trackid", track_ids) \
                        .execute()
                    fp_map = {r["trackid"]: r["filepath"] for r in (resp.data or []) if r.get("filepath")}
                    filepaths = [(tid, fp_map[tid]) for tid in track_ids if tid in fp_map]
                    all_filepaths.extend([fp for _, fp in filepaths])
                except Exception as e:
                    logger.warning(f"Failed to fetch filepaths for playlist {pid}: {e}")

            playlist_data.append({
                "name": playlist["name"],
                "filepaths": filepaths,  # list of (trackid, filepath)
            })

        if not all_filepaths:
            raise HTTPException(status_code=400, detail="No audio files found for the given playlists")

        # Find common parent directory
        common_parent = os.path.commonpath(all_filepaths)
        if os.path.isfile(common_parent):
            common_parent = os.path.dirname(common_parent)
        output_dir = os.path.join(common_parent, "Update")

        folders_created = 0
        files_copied = 0
        errors = []

        for pdata in playlist_data:
            folder_path = os.path.join(output_dir, pdata["name"])
            try:
                os.makedirs(folder_path, exist_ok=True)
                folders_created += 1
            except Exception as e:
                errors.append(f"Failed to create folder '{pdata['name']}': {e}")
                continue

            for tid, fpath in pdata["filepaths"]:
                try:
                    if os.path.isfile(fpath):
                        shutil.copy2(fpath, folder_path)
                        files_copied += 1
                    else:
                        errors.append(f"File not found: {fpath}")
                except Exception as e:
                    errors.append(f"Failed to copy {fpath}: {e}")

        return {
            "output_directory": output_dir,
            "folders_created": folders_created,
            "files_copied": files_copied,
            "errors": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export folders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class OrganizeRequest(BaseModel):
    query: str


@router.post("/organize")
async def organize_playlists(req: OrganizeRequest):
    """Use the LLM to interpret a natural-language playlist command and execute it."""
    try:
        db = _get_db()

        # 1. Existing playlists
        existing_playlists = await db.get_playlist_tree()

        # 2. Track summary (total count + sample of tags/vibes)
        track_pool = await db.get_track_pool()
        total_tracks = len(track_pool)
        sample_tags: set = set()
        sample_vibes: set = set()
        for t in track_pool[:200]:
            for tag in (t.get("semantic_tags") or []):
                sample_tags.add(tag)
            for v in (t.get("vibe") or []):
                sample_vibes.add(v)
        track_summary = {
            "total_tracks": total_tracks,
            "sample_tags": list(sample_tags),
            "sample_vibes": list(sample_vibes),
        }

        # 3. Interpret the command via LLM
        from backend.llm_interpreter import SemanticInterpreter
        interp = SemanticInterpreter(supabase_client=db.client)
        await interp.initialize()
        actions = await interp.interpret_playlist_command(
            req.query, existing_playlists, track_summary
        )

        # 4. Execute each action
        playlists_created = 0
        tracks_assigned = 0
        results = []

        for action in actions:
            tool = action["tool"]
            args = action.get("args", {})

            if tool == "create_playlist":
                result = await _execute_create_playlist(db, args)
                playlists_created += 1
                tracks_assigned += result.get("tracks_added", 0)
                results.append(result)

            elif tool == "add_to_playlist":
                result = await _execute_add_to_playlist(
                    db, args, existing_playlists
                )
                tracks_assigned += result.get("tracks_added", 0)
                results.append(result)

            elif tool == "organize_all":
                result = await _execute_organize_all(db, args)
                playlists_created += result.get("playlists_created", 0)
                tracks_assigned += result.get("tracks_assigned", 0)
                results.append(result)

            elif tool == "suggest_for_playlist":
                result = await _execute_suggest_for_playlist(
                    db, args, existing_playlists
                )
                results.append(result)

        return {
            "message": f"Created {playlists_created} folders, assigned {tracks_assigned} tracks",
            "actions": results,
            "playlists_created": playlists_created,
            "tracks_assigned": tracks_assigned,
        }

    except Exception as e:
        logger.error(f"Organize playlists error: {e}")
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


@router.put("/{playlist_id}/rename")
async def rename_playlist(playlist_id: str, req: RenamePlaylistRequest):
    """Rename a playlist."""
    try:
        db = _get_db()
        success = await db.rename_playlist(playlist_id, req.name)
        if not success:
            raise HTTPException(status_code=404, detail="Playlist not found or rename failed")
        return {"id": playlist_id, "name": req.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rename playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{playlist_id}/add-tracks")
async def add_tracks(playlist_id: str, req: AddTracksRequest):
    """Add tracks to a playlist without removing existing ones."""
    try:
        db = _get_db()
        success = await db.add_tracks_to_playlist(playlist_id, req.track_ids)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add tracks")
        return {"id": playlist_id, "tracks_added": len(req.track_ids)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add tracks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{playlist_id}/remove-tracks")
async def remove_tracks(playlist_id: str, req: RemoveTracksRequest):
    """Remove specific tracks from a playlist."""
    try:
        db = _get_db()
        success = await db.remove_tracks_from_playlist(playlist_id, req.track_ids)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove tracks")
        return {"id": playlist_id, "tracks_removed": len(req.track_ids)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove tracks error: {e}")
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


# ── Organize helpers ─────────────────────────────────────────────────────────


async def _query_tracks_by_criteria(db, criteria: dict, limit: int = 40) -> List[dict]:
    """Query tracks matching a criteria dict from the LLM tool call."""
    structured_query: Dict[str, Any] = {}
    if criteria.get("genres"):
        structured_query["semantic_tags"] = criteria["genres"]
    if criteria.get("vibes"):
        structured_query["vibes"] = criteria["vibes"]
    if criteria.get("energy_range"):
        structured_query["energy_range"] = criteria["energy_range"]
    if criteria.get("bpm_range"):
        structured_query["bpm_range"] = criteria["bpm_range"]

    max_tracks = criteria.get("max_tracks", limit)
    tracks = await db.get_tracks_by_semantic_filter(structured_query, limit=max_tracks)
    return tracks


async def _execute_create_playlist(db, args: dict) -> dict:
    """Create a playlist and populate it with tracks matching criteria."""
    name = args.get("name", "Untitled")
    criteria = args.get("criteria", {})
    playlist_id = str(uuid.uuid4())

    await db.create_playlist(playlist_id, name, source="llm-organize")

    tracks = await _query_tracks_by_criteria(db, criteria)
    track_ids = []
    for t in tracks:
        tid = t.trackid if hasattr(t, "trackid") else t.get("trackid")
        if tid:
            track_ids.append(str(tid))

    if track_ids:
        await db.add_tracks_to_playlist(playlist_id, track_ids)

    return {
        "action": "create_playlist",
        "playlist_id": playlist_id,
        "playlist_name": name,
        "tracks_added": len(track_ids),
    }


async def _execute_add_to_playlist(
    db, args: dict, existing_playlists: list
) -> dict:
    """Add tracks matching criteria to an existing playlist (found by name)."""
    playlist_name = args.get("playlist_name", "")
    criteria = args.get("criteria", {})

    # Find playlist by name (case-insensitive)
    target = None
    for p in existing_playlists:
        if p.get("name", "").lower() == playlist_name.lower():
            target = p
            break

    if not target:
        return {
            "action": "add_to_playlist",
            "error": f"Playlist '{playlist_name}' not found",
            "tracks_added": 0,
        }

    tracks = await _query_tracks_by_criteria(db, criteria)
    track_ids = []
    for t in tracks:
        tid = t.trackid if hasattr(t, "trackid") else t.get("trackid")
        if tid:
            track_ids.append(str(tid))

    if track_ids:
        await db.add_tracks_to_playlist(target["id"], track_ids)

    return {
        "action": "add_to_playlist",
        "playlist_id": target["id"],
        "playlist_name": playlist_name,
        "tracks_added": len(track_ids),
    }


async def _execute_organize_all(db, args: dict) -> dict:
    """Create multiple playlists and distribute tracks, avoiding duplicates."""
    playlist_defs = args.get("playlists", [])
    max_per = args.get("max_per_playlist", 40)
    used_tracks: set = set()
    total_created = 0
    total_assigned = 0
    details = []

    for pdef in playlist_defs:
        name = pdef.get("name", "Untitled")
        criteria = pdef.get("criteria", {})
        playlist_id = str(uuid.uuid4())

        await db.create_playlist(playlist_id, name, source="llm-organize")
        total_created += 1

        limit = criteria.get("max_tracks", max_per)
        tracks = await _query_tracks_by_criteria(db, criteria, limit=limit + len(used_tracks))

        track_ids = []
        for t in tracks:
            tid = t.trackid if hasattr(t, "trackid") else t.get("trackid")
            tid = str(tid) if tid else None
            if tid and tid not in used_tracks:
                track_ids.append(tid)
                used_tracks.add(tid)
                if len(track_ids) >= limit:
                    break

        if track_ids:
            await db.add_tracks_to_playlist(playlist_id, track_ids)
            total_assigned += len(track_ids)

        details.append({
            "playlist_id": playlist_id,
            "playlist_name": name,
            "tracks_added": len(track_ids),
        })

    return {
        "action": "organize_all",
        "playlists_created": total_created,
        "tracks_assigned": total_assigned,
        "details": details,
    }


async def _execute_suggest_for_playlist(
    db, args: dict, existing_playlists: list
) -> dict:
    """Suggest unassigned tracks that would fit a playlist."""
    playlist_name = args.get("playlist_name", "")
    count = args.get("count", 10)

    # Find playlist by name
    target = None
    for p in existing_playlists:
        if p.get("name", "").lower() == playlist_name.lower():
            target = p
            break

    if not target:
        return {
            "action": "suggest_for_playlist",
            "error": f"Playlist '{playlist_name}' not found",
            "suggestions": [],
        }

    # Get existing tracks in playlist
    playlist_data = await db.get_playlist(target["id"])
    existing_tracks = playlist_data.get("tracks", []) if playlist_data else []
    existing_ids = {t.get("trackid") for t in existing_tracks}

    # Build a search prompt from the playlist's current tracks
    prompt = _build_suggest_prompt(existing_tracks)

    from backend.llm_interpreter import SemanticInterpreter, InterpretationContext
    interp = SemanticInterpreter(supabase_client=db.client)
    await interp.initialize()

    context = None
    if existing_tracks:
        context = InterpretationContext(current_track=existing_tracks[-1])

    params = await interp.interpret(prompt, context=context)

    # tag_scores / vibe_scores are dicts {name: confidence}
    tag_scores = params.get("tag_scores") or {}
    vibe_scores = params.get("vibe_scores") or {}
    structured_query = {
        "semantic_tags": list(tag_scores.keys()) if isinstance(tag_scores, dict) else list(tag_scores),
        "vibes": list(vibe_scores.keys()) if isinstance(vibe_scores, dict) else list(vibe_scores),
        "bpm_range": params.get("bpm_range"),
        "energy_range": params.get("energy_range"),
    }

    candidates = await db.get_tracks_by_semantic_filter(
        structured_query, limit=count + len(existing_ids)
    )

    suggestions = []
    for track in candidates:
        tid = track.trackid if hasattr(track, "trackid") else track.get("trackid")
        tid = str(tid) if tid else None
        if tid and tid not in existing_ids:
            suggestions.append({
                "trackid": tid,
                "title": track.title if hasattr(track, "title") else track.get("title"),
                "artist": track.artist if hasattr(track, "artist") else track.get("artist"),
                "bpm": track.bpm if hasattr(track, "bpm") else track.get("bpm"),
                "energy": track.energy if hasattr(track, "energy") else track.get("energy"),
            })
            if len(suggestions) >= count:
                break

    return {
        "action": "suggest_for_playlist",
        "playlist_name": playlist_name,
        "suggestions": suggestions,
    }
