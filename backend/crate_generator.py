import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


CAMELOT_MAP = {
    "C": "8B", "Am": "8A", "Cm": "5A", "C#": "3B", "C#m": "12A",
    "Db": "3B", "Dbm": "12A", "D": "10B", "Dm": "7A", "D#": "5B",
    "D#m": "2A", "Eb": "5B", "Ebm": "2A", "E": "12B", "Em": "9A",
    "F": "7B", "Fm": "4A", "F#": "2B", "F#m": "11A", "Gb": "2B",
    "Gbm": "11A", "G": "9B", "Gm": "6A", "G#": "4B", "G#m": "1A",
    "Ab": "4B", "Abm": "1A", "A": "11B", "A#": "6B", "A#m": "3A",
    "Bb": "6B", "Bbm": "3A", "B": "1B", "Bm": "10A",
}


@dataclass
class CrateSpec:
    """Specification for a crate to generate."""

    session_id: str
    parent_crate_id: Optional[str] = None
    genres: List[Dict[str, float]] = field(default_factory=list)   # [{name, confidence}]
    vibes: List[Dict[str, float]] = field(default_factory=list)
    energy_range: Optional[List[float]] = None  # [min, max] 1-10
    bpm_range: Optional[List[float]] = None
    track_count: int = 5
    direction: Optional[str] = None       # energy_up, energy_down, genre_shift, bridge, maintain
    direction_desc: Optional[str] = None


def _attr_or_key(obj, name, default=None):
    """Read a value from an object attribute or dict key."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _track_to_dict(t) -> Dict:
    """Normalise a TrackMetadata object or raw dict into a plain dict."""
    return {
        "trackid": _attr_or_key(t, "trackid"),
        "title": _attr_or_key(t, "title"),
        "artist": _attr_or_key(t, "artist"),
        "bpm": _attr_or_key(t, "bpm"),
        "key": _attr_or_key(t, "key"),
        "energy": _attr_or_key(t, "energy"),
        "semantic_tags": _attr_or_key(t, "semantic_tags", []),
        "vibe_descriptors": _attr_or_key(t, "vibe_descriptors", []),
    }


def key_compatibility(key_a: Optional[str], key_b: Optional[str]) -> float:
    """Calculate Camelot Wheel key compatibility between two musical keys.

    Returns a float 0.0-1.0:
        1.0  — same Camelot code
        0.85 — adjacent number, same letter (e.g. 8A -> 7A)
        0.8  — same number, different letter (e.g. 8A -> 8B)
        0.5  — +-2 semitone distance
        0.2  — everything else
    """
    if not key_a or not key_b:
        return 0.5

    ca = CAMELOT_MAP.get(key_a.strip())
    cb = CAMELOT_MAP.get(key_b.strip())
    if not ca or not cb:
        return 0.3

    num_a, letter_a = int(ca[:-1]), ca[-1]
    num_b, letter_b = int(cb[:-1]), cb[-1]

    if ca == cb:
        return 1.0
    if num_a == num_b and letter_a != letter_b:
        return 0.8
    diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if diff == 1 and letter_a == letter_b:
        return 0.85
    if diff <= 2:
        return 0.5
    return 0.2


class CrateGenerator:
    """Generates cohesive crates (~5 tracks) using BPM, key and energy heuristics."""

    def __init__(self, db_manager):
        self.db = db_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_crate(
        self,
        spec: CrateSpec,
        parent_tracks: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate a crate of cohesive tracks.

        Algorithm
        ---------
        1. Build a search query from the CrateSpec.
        2. Fetch candidates via the existing semantic filter pipeline.
        3. If *parent_tracks* is provided, narrow candidates to those that
           transition smoothly from the last parent track.
        4. Greedy selection: pick a seed then iteratively add the track
           that maximises coherence with the growing set.
        5. Order the selected tracks for best DJ flow (nearest-neighbour TSP).
        6. Compute aggregate crate metadata and a human-readable label.
        """
        search_query = self._build_search_query(spec)

        # Fetch more candidates than needed so the selector has room
        candidates = await self._get_candidates(search_query, limit=spec.track_count * 8)

        if not candidates:
            return {"tracks": [], "metadata": {}, "label": "Empty crate"}

        # Narrow to transition-compatible tracks when chaining crates
        if parent_tracks:
            last_parent = parent_tracks[-1]
            candidates = self._filter_transition_compat(candidates, last_parent)

        selected = self._greedy_select(candidates, spec.track_count, parent_tracks)
        ordered = self._order_for_flow(selected, parent_tracks)
        metadata = self._compute_metadata(ordered)
        label = self._generate_label(metadata, spec)

        return {
            "tracks": ordered,
            "metadata": metadata,
            "label": label,
        }

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_search_query(spec: CrateSpec) -> Dict:
        """Convert a CrateSpec into a dict accepted by ``db.get_tracks_by_semantic_filter``."""
        query: Dict[str, Any] = {}
        if spec.genres:
            query["semantic_tags"] = [
                g["name"] for g in spec.genres if g.get("confidence", 0) > 0.3
            ]
        if spec.vibes:
            query["vibes"] = [
                v["name"] for v in spec.vibes if v.get("confidence", 0) > 0.3
            ]
        if spec.energy_range:
            query["energy_range"] = spec.energy_range
        if spec.bpm_range:
            query["bpm_range"] = spec.bpm_range
        return query

    # ------------------------------------------------------------------
    # Candidate fetching
    # ------------------------------------------------------------------

    async def _get_candidates(self, search_query: Dict, limit: int = 40) -> List[Dict]:
        """Fetch candidate tracks from DB using the semantic filter."""
        try:
            results = await self.db.get_tracks_by_semantic_filter(search_query, limit=limit)
            return [_track_to_dict(t) for t in results]
        except Exception as e:
            logger.error("Failed to get candidates: %s", e)
            return []

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_transition_compat(
        candidates: List[Dict],
        anchor: Dict,
        min_compat: float = 0.4,
    ) -> List[Dict]:
        """Keep only candidates that can transition smoothly from *anchor*."""
        result = []
        for c in candidates:
            bpm_diff = abs((c.get("bpm") or 128) - (anchor.get("bpm") or 128))
            kc = key_compatibility(c.get("key"), anchor.get("key"))
            if bpm_diff <= 12 and kc >= min_compat:
                result.append(c)
        # Fall back to top candidates if the filter is too aggressive
        return result if result else candidates[:20]

    # ------------------------------------------------------------------
    # Greedy selection
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy_select(
        candidates: List[Dict],
        count: int,
        parent_tracks: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Pick *count* tracks from *candidates* maximising internal coherence.

        Scoring weights per step:
            40 % search relevance (position in the candidate list)
            25 % BPM compatibility with the previously-selected track
            25 % key compatibility with the previously-selected track
            10 % energy arc (slight upward trend preferred)

        A heavy penalty is applied when adding a track would push the overall
        BPM spread beyond 6 BPM.
        """
        if not candidates:
            return []
        if len(candidates) <= count:
            return list(candidates)

        selected: List[Dict] = [candidates[0]]  # seed = top search result
        remaining = list(candidates[1:])
        n_candidates = len(candidates)

        for _ in range(count - 1):
            if not remaining:
                break

            last = selected[-1]
            best_score = -1.0
            best_idx = 0

            for i, c in enumerate(remaining):
                # Relevance: inverse of original position
                relevance = 1.0 - (candidates.index(c) / max(n_candidates, 1))

                # BPM compatibility
                bpm_diff = abs((c.get("bpm") or 128) - (last.get("bpm") or 128))
                bpm_score = max(0.0, 1.0 - bpm_diff / 15)

                # Key compatibility
                key_score = key_compatibility(c.get("key"), last.get("key"))

                # Energy arc: slight upward trend within a crate (energy is 1-10)
                last_energy = last.get("energy") or 5
                c_energy = c.get("energy") or 5
                energy_arc = max(0.0, min(1.0, 0.5 + (c_energy - last_energy) / 10))

                score = (
                    relevance * 0.40
                    + bpm_score * 0.25
                    + key_score * 0.25
                    + energy_arc * 0.10
                )

                # Penalise if BPM spread within the crate would exceed 6 BPM
                all_bpms = [s.get("bpm") or 128 for s in selected] + [c.get("bpm") or 128]
                if max(all_bpms) - min(all_bpms) > 6:
                    score *= 0.3

                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    # ------------------------------------------------------------------
    # Ordering (nearest-neighbour TSP)
    # ------------------------------------------------------------------

    def _order_for_flow(
        self,
        tracks: List[Dict],
        parent_tracks: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Order tracks for best DJ flow using a nearest-neighbour heuristic.

        The starting track is chosen as:
        - The one most compatible with the parent crate's last track, or
        - The lowest-energy track when there is no parent.
        """
        if len(tracks) <= 2:
            return tracks

        if parent_tracks:
            anchor = parent_tracks[-1]
            start_idx = max(
                range(len(tracks)),
                key=lambda i: self._transition_score(anchor, tracks[i]),
            )
        else:
            start_idx = min(
                range(len(tracks)),
                key=lambda i: tracks[i].get("energy") or 0.5,
            )

        ordered = [tracks[start_idx]]
        remaining = [t for i, t in enumerate(tracks) if i != start_idx]

        while remaining:
            last = ordered[-1]
            best_idx = max(
                range(len(remaining)),
                key=lambda i: self._transition_score(last, remaining[i]),
            )
            ordered.append(remaining.pop(best_idx))

        return ordered

    @staticmethod
    def _transition_score(a: Dict, b: Dict) -> float:
        """Score how well track *b* follows track *a*.

        Weighted 40 % key / 30 % BPM / 30 % energy smoothness.
        """
        key_score = key_compatibility(a.get("key"), b.get("key"))
        bpm_diff = abs((a.get("bpm") or 128) - (b.get("bpm") or 128))
        bpm_score = max(0.0, 1.0 - bpm_diff / 15)
        energy_diff = abs((a.get("energy") or 5) - (b.get("energy") or 5))
        energy_smooth = max(0.0, 1.0 - energy_diff / 5)  # 5-point diff → 0
        return key_score * 0.4 + bpm_score * 0.3 + energy_smooth * 0.3

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metadata(tracks: List[Dict]) -> Dict:
        """Compute aggregate metadata for a crate."""
        if not tracks:
            return {}

        bpms = [t["bpm"] for t in tracks if t.get("bpm")]
        energies = [t["energy"] for t in tracks if t.get("energy") is not None]
        keys = [t["key"] for t in tracks if t.get("key")]

        all_tags: List[str] = []
        for t in tracks:
            all_tags.extend(t.get("semantic_tags") or [])

        # Dominant key
        key_counts: Dict[str, int] = {}
        for k in keys:
            key_counts[k] = key_counts.get(k, 0) + 1
        dominant_key = max(key_counts, key=key_counts.get) if key_counts else None

        # Top tags
        tag_counts: Dict[str, int] = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        dominant_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3]

        return {
            "avg_bpm": round(sum(bpms) / len(bpms), 1) if bpms else None,
            "avg_energy": round(sum(energies) / len(energies), 2) if energies else None,
            "bpm_range": [min(bpms), max(bpms)] if bpms else None,
            "energy_range": [round(min(energies), 2), round(max(energies), 2)] if energies else None,
            "dominant_key": dominant_key,
            "dominant_tags": dominant_tags,
            "track_count": len(tracks),
        }

    @staticmethod
    def _generate_label(metadata: Dict, spec: CrateSpec) -> str:
        """Generate a human-readable label for the crate."""
        parts: List[str] = []
        if metadata.get("dominant_tags"):
            parts.append(metadata["dominant_tags"][0].title())
        if metadata.get("avg_bpm"):
            parts.append(f"{metadata['avg_bpm']} BPM")
        if spec.direction:
            direction_labels = {
                "energy_up": "High Energy",
                "energy_down": "Downtempo",
                "genre_shift": "Genre Shift",
                "bridge": "Bridge",
                "maintain": "Steady",
            }
            label = direction_labels.get(spec.direction)
            if label:
                parts.append(label)
        return " / ".join(parts) or "Crate"


# ======================================================================
# Bridge Analyzer
# ======================================================================


class BridgeAnalyzer:
    """Analyses gaps between crates and suggests bridge tracks."""

    def __init__(self, db_manager):
        self.db = db_manager

    async def analyze_bridge(
        self,
        crate_a_tracks: List[Dict],
        crate_b_tracks: List[Dict],
    ) -> Dict:
        """Analyse the gap between two adjacent crates.

        Returns a dict with *gap_score* (1.0 = seamless, 0.0 = jarring),
        whether a bridge is needed, and suggested bridge tracks when it is.
        """
        if not crate_a_tracks or not crate_b_tracks:
            return {"gap_score": 0.0, "needs_bridge": True, "bridge_tracks": []}

        last_a = crate_a_tracks[-1]
        first_b = crate_b_tracks[0]

        bpm_gap = abs((last_a.get("bpm") or 128) - (first_b.get("bpm") or 128))
        energy_gap = abs((last_a.get("energy") or 0.5) - (first_b.get("energy") or 0.5))
        key_compat = key_compatibility(last_a.get("key"), first_b.get("key"))

        # Gap score: 1.0 = perfect transition, 0.0 = terrible
        gap_score = 1.0
        if bpm_gap > 8:
            gap_score -= 0.3 * min(1, (bpm_gap - 8) / 20)
        if energy_gap > 0.25:
            gap_score -= 0.3 * min(1, (energy_gap - 0.25) / 0.75)
        if key_compat < 0.5:
            gap_score -= 0.2
        gap_score = max(0.0, gap_score)

        needs_bridge = gap_score < 0.6
        bridge_tracks: List[Dict] = []

        if needs_bridge:
            bridge_tracks = await self._find_bridge_tracks(last_a, first_b)

        return {
            "gap_score": round(gap_score, 3),
            "bpm_gap": round(bpm_gap, 1),
            "energy_gap": round(energy_gap, 2),
            "key_compatible": key_compat >= 0.5,
            "needs_bridge": needs_bridge,
            "bridge_tracks": bridge_tracks,
        }

    async def _find_bridge_tracks(
        self,
        track_a: Dict,
        track_b: Dict,
        limit: int = 3,
    ) -> List[Dict]:
        """Find tracks that bridge the gap between two endpoint tracks.

        Targets the midpoint BPM and energy, then ranks by dual key
        compatibility (must be compatible with both endpoints).
        """
        bpm_a = track_a.get("bpm") or 128
        bpm_b = track_b.get("bpm") or 128
        energy_a = track_a.get("energy") or 0.5
        energy_b = track_b.get("energy") or 0.5

        mid_bpm = (bpm_a + bpm_b) / 2
        bpm_margin = max(4, abs(bpm_a - bpm_b) / 2 + 2)

        mid_energy = (energy_a + energy_b) / 2
        energy_margin = max(0.15, abs(energy_a - energy_b) / 2 + 0.1)

        search_query = {
            "bpm_range": [mid_bpm - bpm_margin, mid_bpm + bpm_margin],
            "energy_range": [
                max(0.0, mid_energy - energy_margin),
                min(1.0, mid_energy + energy_margin),
            ],
        }

        try:
            results = await self.db.get_tracks_by_semantic_filter(search_query, limit=20)

            scored: List[Tuple[float, Dict]] = []
            for t in results:
                t_dict = _track_to_dict(t)

                compat_a = key_compatibility(t_dict.get("key"), track_a.get("key"))
                compat_b = key_compatibility(t_dict.get("key"), track_b.get("key"))

                # Must be reasonably compatible with both endpoints
                if compat_a < 0.4 or compat_b < 0.4:
                    continue

                score = (compat_a + compat_b) / 2
                scored.append((score, t_dict))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:limit]]
        except Exception as e:
            logger.error("Bridge track search failed: %s", e)
            return []
