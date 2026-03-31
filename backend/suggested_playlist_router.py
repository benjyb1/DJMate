"""
API router for suggested playlists — auto-generated DJ-ready track groupings.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from backend.tenant import get_tenant_db
from backend.data.db_interface import DatabaseManager
from backend.suggested_playlist_generator import generate_playlists, name_playlist_with_llm

logger = logging.getLogger(__name__)
router = APIRouter()


async def _load_embeddings_and_bpms(db: DatabaseManager):
    """Load all track embeddings and BPMs from the database."""
    if not db.client:
        raise HTTPException(status_code=503, detail="Database not available")

    # Fetch embeddings + bpm
    resp = db.client.table("tracks").select(
        "trackid, embedding, bpm"
    ).not_.is_("embedding", "null").execute()

    if not resp.data:
        raise HTTPException(status_code=404, detail="No tracks with embeddings found")

    track_ids = []
    embeddings = []
    bpms = []

    for row in resp.data:
        emb = row.get("embedding")
        if emb is None:
            continue
        # Parse embedding (may be JSON string or list)
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(emb, list) or len(emb) == 0:
            continue

        track_ids.append(row["trackid"])
        embeddings.append(emb)
        bpms.append(row.get("bpm") or 120.0)

    if len(track_ids) < 7:
        raise HTTPException(status_code=400, detail="Need at least 7 tracks with embeddings")

    # Build normalised embedding matrix
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    return track_ids, emb_matrix, bpms


async def _load_track_labels(db: DatabaseManager, track_ids: List[int]) -> Dict[int, Dict]:
    """Load labels (tags, vibes, energy) for a set of tracks."""
    if not db.client:
        return {}

    resp = db.client.table("track_labels").select(
        "trackid, semantic_tags, vibe, energy"
    ).in_("trackid", track_ids).execute()

    labels = {}
    if resp.data:
        for row in resp.data:
            labels[row["trackid"]] = row
    return labels


async def _load_track_metadata(db: DatabaseManager, track_ids: List[int]) -> Dict[int, Dict]:
    """Load basic metadata for a set of tracks."""
    if not db.client:
        return {}

    resp = db.client.table("tracks").select(
        "trackid, title, artist, bpm, key, filepath, album_art_url"
    ).in_("trackid", track_ids).execute()

    meta = {}
    if resp.data:
        for row in resp.data:
            meta[row["trackid"]] = row
    return meta


def _get_llm_providers() -> list:
    """Build the list of LLM providers for naming (reuses the same pattern as SemanticInterpreter)."""
    import os
    providers = []
    try:
        from openai import AsyncOpenAI
        if os.getenv("GROQ_API_KEY"):
            providers.append({
                "name": "Groq",
                "client": AsyncOpenAI(
                    api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                ),
                "model": "llama-3.3-70b-versatile",
            })
        if os.getenv("GEMINI_API_KEY"):
            providers.append({
                "name": "Gemini",
                "client": AsyncOpenAI(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                "model": "gemini-2.5-flash",
            })
        if os.getenv("OPENAI_API_KEY"):
            providers.append({
                "name": "OpenAI",
                "client": AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                "model": "gpt-4o-mini",
            })
    except ImportError:
        logger.warning("openai package not installed — LLM naming unavailable")
    return providers


@router.post("/generate")
async def generate_suggested_playlists(db: DatabaseManager = Depends(get_tenant_db)):
    """Generate a fresh batch of suggested playlists."""
    logger.info("Generating suggested playlists...")

    # 1. Load data
    track_ids, emb_matrix, bpms = await _load_embeddings_and_bpms(db)
    logger.info(f"Loaded {len(track_ids)} tracks with embeddings")

    # 2. Run clustering algorithm
    raw_playlists = generate_playlists(track_ids, emb_matrix, bpms)
    if not raw_playlists:
        raise HTTPException(status_code=400, detail="Could not generate playlists from current library")

    # 3. Load labels for naming
    all_playlist_track_ids = set()
    for pl in raw_playlists:
        all_playlist_track_ids.update(pl["track_ids"])
    labels = await _load_track_labels(db, list(all_playlist_track_ids))
    meta = await _load_track_metadata(db, list(all_playlist_track_ids))

    # 4. Name each playlist with LLM
    providers = _get_llm_providers()
    results = []

    for pl in raw_playlists:
        # Build enriched track data for naming
        tracks_data = []
        for tid in pl["track_ids"]:
            row = {**(meta.get(tid, {})), **(labels.get(tid, {}))}
            tracks_data.append(row)

        name = await name_playlist_with_llm(tracks_data, providers)

        # 5. Store in DB
        try:
            insert_resp = db.client.table("suggested_playlists").insert({
                "name": name,
                "seed_trackid": pl["seed_trackid"],
                "track_ids": pl["track_ids"],
            }).execute()
        except Exception as e:
            logger.error(f"Failed to store playlist '{name}': {e}")
            continue

        # Build response with track details
        bpm_vals = [meta.get(tid, {}).get("bpm") for tid in pl["track_ids"]]
        bpm_vals = [b for b in bpm_vals if b is not None]

        results.append({
            "id": insert_resp.data[0]["id"] if insert_resp.data else None,
            "name": name,
            "seed_trackid": pl["seed_trackid"],
            "track_count": len(pl["track_ids"]),
            "bpm_range": [round(min(bpm_vals)), round(max(bpm_vals))] if bpm_vals else None,
            "tracks": [
                {
                    "trackid": tid,
                    "title": meta.get(tid, {}).get("title", ""),
                    "artist": meta.get(tid, {}).get("artist", ""),
                    "bpm": meta.get(tid, {}).get("bpm"),
                    "key": meta.get(tid, {}).get("key"),
                    "album_art_url": meta.get(tid, {}).get("album_art_url"),
                    "energy": labels.get(tid, {}).get("energy"),
                }
                for tid in pl["track_ids"]
            ],
        })

    logger.info(f"Generated and stored {len(results)} suggested playlists")
    return {"playlists": results, "count": len(results)}


@router.get("")
async def get_suggested_playlists(db: DatabaseManager = Depends(get_tenant_db)):
    """Fetch the most recent generation of suggested playlists."""
    if not db.client:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get the latest generated_at timestamp
    latest_resp = db.client.table("suggested_playlists").select(
        "generated_at"
    ).order("generated_at", desc=True).limit(1).execute()

    if not latest_resp.data:
        return {"playlists": [], "count": 0}

    latest_ts = latest_resp.data[0]["generated_at"]

    # Fetch all playlists from that generation
    resp = db.client.table("suggested_playlists").select("*").eq(
        "generated_at", latest_ts
    ).execute()

    if not resp.data:
        # Fallback: just get the most recent N playlists
        resp = db.client.table("suggested_playlists").select("*").order(
            "generated_at", desc=True
        ).limit(15).execute()

    if not resp.data:
        return {"playlists": [], "count": 0}

    # Load track metadata for all playlists
    all_track_ids = set()
    for pl in resp.data:
        all_track_ids.update(pl["track_ids"])

    meta = await _load_track_metadata(db, list(all_track_ids))
    labels = await _load_track_labels(db, list(all_track_ids))

    playlists = []
    for pl in resp.data:
        bpm_vals = [meta.get(tid, {}).get("bpm") for tid in pl["track_ids"]]
        bpm_vals = [b for b in bpm_vals if b is not None]

        playlists.append({
            "id": pl["id"],
            "name": pl["name"],
            "seed_trackid": pl["seed_trackid"],
            "track_count": len(pl["track_ids"]),
            "bpm_range": [round(min(bpm_vals)), round(max(bpm_vals))] if bpm_vals else None,
            "tracks": [
                {
                    "trackid": tid,
                    "title": meta.get(tid, {}).get("title", ""),
                    "artist": meta.get(tid, {}).get("artist", ""),
                    "bpm": meta.get(tid, {}).get("bpm"),
                    "key": meta.get(tid, {}).get("key"),
                    "album_art_url": meta.get(tid, {}).get("album_art_url"),
                    "energy": labels.get(tid, {}).get("energy"),
                }
                for tid in pl["track_ids"]
            ],
            "generated_at": pl["generated_at"],
        })

    return {"playlists": playlists, "count": len(playlists)}
