"""
rekordbox_parser.py — Parse Rekordbox XML and match tracks to DJMate DB.

Rekordbox XML format:
- Root: DJ_PLAYLISTS
- COLLECTION/TRACK elements with @TrackID, @Name, @Artist, @TotalTime, @AverageBpm, @Tonality, @Location
- PLAYLISTS/NODE @Type="1" (folder) or @Type="0" (playlist)
  - TRACK @Key references COLLECTION @TrackID
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
import logging
import re
from urllib.parse import unquote

logger = logging.getLogger(__name__)


def parse_xml(file_content: str) -> dict:
    """
    Parse a Rekordbox XML string.

    Returns:
        dict with:
          - collection: list of track dicts (trackid, name, artist, total_time, bpm, key, location)
          - playlists: list of {name, track_keys}
    """
    root = ET.fromstring(file_content)

    # Parse COLLECTION
    collection = []
    collection_el = root.find("COLLECTION")
    if collection_el is not None:
        for track_el in collection_el.findall("TRACK"):
            track = {
                "trackid": track_el.get("TrackID"),
                "name": track_el.get("Name", ""),
                "artist": track_el.get("Artist", ""),
                "total_time": _safe_int(track_el.get("TotalTime")),
                "bpm": _safe_float(track_el.get("AverageBpm")),
                "key": track_el.get("Tonality", ""),
                "location": unquote(track_el.get("Location", "")),
            }
            collection.append(track)

    # Parse PLAYLISTS (recursive NODE traversal)
    playlists = []
    playlists_el = root.find("PLAYLISTS")
    if playlists_el is not None:
        _walk_playlist_nodes(playlists_el, playlists)

    return {"collection": collection, "playlists": playlists}


def match_tracks(parsed_tracks: List[dict], db_tracks: List[dict]) -> list:
    """
    Fuzzy match parsed XML tracks to DB tracks using token overlap scoring
    on title + artist.

    Args:
        parsed_tracks: list of dicts from parse_xml (keys: name, artist)
        db_tracks: list of dicts from DB (keys: title, artist, trackid)

    Returns:
        list of {xml_track, db_trackid, confidence}
    """
    results = []

    # Pre-tokenize DB tracks
    db_tokens = []
    for dt in db_tracks:
        tokens = _tokenize(f"{dt.get('title', '')} {dt.get('artist', '')}")
        db_tokens.append((dt, tokens))

    for xml_track in parsed_tracks:
        query_tokens = _tokenize(f"{xml_track.get('name', '')} {xml_track.get('artist', '')}")
        if not query_tokens:
            results.append({"xml_track": xml_track, "db_trackid": None, "confidence": 0.0})
            continue

        best_match = None
        best_score = 0.0

        for dt, dt_tokens in db_tokens:
            if not dt_tokens:
                continue
            # Token overlap: Jaccard-like but weighted toward query coverage
            overlap = len(query_tokens & dt_tokens)
            union = len(query_tokens | dt_tokens)
            if union == 0:
                continue
            score = overlap / union
            if score > best_score:
                best_score = score
                best_match = dt

        results.append({
            "xml_track": xml_track,
            "db_trackid": best_match.get("trackid") if best_match and best_score > 0.3 else None,
            "confidence": round(best_score, 3),
        })

    return results


def generate_export_xml(playlist_name: str, tracks: List[dict]) -> str:
    """
    Generate a valid Rekordbox XML string from a playlist.

    Args:
        playlist_name: name of the playlist
        tracks: list of track dicts with keys: trackid, title, artist, bpm, key, duration

    Returns:
        XML string in Rekordbox format
    """
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")

    # PRODUCT element
    product = ET.SubElement(root, "PRODUCT", Name="DJMate", Version="1.0.0")

    # COLLECTION
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))
    for i, track in enumerate(tracks):
        track_id = str(i + 1)
        attrs = {
            "TrackID": track_id,
            "Name": track.get("title", ""),
            "Artist": track.get("artist", ""),
            "TotalTime": str(int(track.get("duration", 0) or 0)),
            "AverageBpm": f"{float(track.get('bpm', 0) or 0):.2f}",
            "Tonality": track.get("key", ""),
        }
        ET.SubElement(collection, "TRACK", **attrs)

    # PLAYLISTS
    playlists = ET.SubElement(root, "PLAYLISTS")
    root_node = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count=str(1))
    playlist_node = ET.SubElement(
        root_node, "NODE",
        Name=playlist_name, Type="0", KeyType="0", Entries=str(len(tracks)),
    )
    for i in range(len(tracks)):
        ET.SubElement(playlist_node, "TRACK", Key=str(i + 1))

    # Serialize
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _walk_playlist_nodes(parent_el: ET.Element, playlists: list):
    """Recursively walk NODE elements, collecting Type='0' playlists."""
    for node in parent_el.findall("NODE"):
        node_type = node.get("Type", "0")
        if node_type == "1":
            # Folder — recurse
            _walk_playlist_nodes(node, playlists)
        else:
            # Playlist
            name = node.get("Name", "Untitled")
            track_keys = [t.get("Key") for t in node.findall("TRACK") if t.get("Key")]
            playlists.append({"name": name, "track_keys": track_keys})


def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split into token set."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return {t for t in text.split() if len(t) > 1}


def _safe_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
