"""
online_learner.py — Online learning system for DJMate tag corrections.

Three layers:
  1. Correction capture  — already handled by tag_router.py → tag_corrections table
  2. Pattern clustering   — find_correction_patterns() groups corrections by embedding neighborhood
  3. Propagation          — propagate() applies confident patterns to auto-tagged tracks

Usage:
  learner = OnlineLearner()
  patterns = await learner.find_correction_patterns()
  result   = await learner.propagate(dry_run=False)
  status   = await learner.get_status()
"""

import os
import json
import logging
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

NEIGHBOR_K = 20                # how many neighbors to check per correction
AGREEMENT_THRESHOLD = 0.50     # fraction of neighbors that must share the wrong tag
CONFIDENCE_PROPAGATE = 0.70    # minimum pattern confidence for propagation
HALF_LIFE_DAYS = 30            # recency weighting half-life
MIN_CORRECTIONS_GLOBAL = 10    # trigger propagation after N total corrections
MIN_CORRECTIONS_LOCAL = 5      # …or N corrections in the same neighborhood
MERGE_COSINE_THRESHOLD = 0.90  # patterns closer than this get merged


class CorrectionPattern:
    """A cluster of similar corrections in embedding space."""

    def __init__(self, center: np.ndarray, radius: float, field: str,
                 from_value: str, to_value: str, confidence: float,
                 correction_ids: list):
        self.center = center
        self.radius = radius
        self.field = field
        self.from_value = from_value
        self.to_value = to_value
        self.confidence = confidence
        self.correction_ids = correction_ids

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "confidence": round(self.confidence, 3),
            "radius": round(self.radius, 4),
            "num_corrections": len(self.correction_ids),
        }


class OnlineLearner:
    """Learns from user tag corrections and propagates confident patterns."""

    def __init__(self):
        from backend.data.db_interface import DatabaseManager
        self._db = DatabaseManager(enable_caching=True, pool_size=10)
        self._patterns: list[CorrectionPattern] = []
        self._last_propagation: datetime | None = None

    # ── Layer 2: Pattern clustering ──────────────────────────────────────────

    async def find_correction_patterns(self) -> list[CorrectionPattern]:
        """
        Cluster recent corrections by embedding neighborhood.

        For each correction:
          1. Find the corrected track's embedding
          2. Find K nearest neighbors
          3. Check if >50% of neighbors share the old (wrong) value
          4. If yes → create a CorrectionPattern
        Then merge overlapping patterns.
        """
        corrections = await self._db.get_recent_corrections(since_days=HALF_LIFE_DAYS * 2)
        if not corrections:
            logger.info("No recent corrections found")
            self._patterns = []
            return []

        # Build embedding matrix from all tracks that have embeddings
        all_tracks = await self._fetch_tracks_with_embeddings()
        if not all_tracks:
            logger.warning("No tracks with embeddings found")
            self._patterns = []
            return []

        from backend.data.embeddings import build_embedding_matrix
        track_ids, emb_matrix = build_embedding_matrix(all_tracks)
        if len(track_ids) == 0:
            self._patterns = []
            return []

        # Index for fast lookup: trackid → row index
        id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

        # Build a lookup for track labels: trackid → label row
        label_lookup = await self._build_label_lookup(track_ids)

        patterns: list[CorrectionPattern] = []

        for corr in corrections:
            cid = corr.get("id")
            tid = corr.get("trackid")
            field = corr.get("field")
            old_value = corr.get("old_value")
            new_value = corr.get("new_value")

            if tid not in id_to_idx:
                continue

            idx = id_to_idx[tid]
            query_vec = emb_matrix[idx].reshape(1, -1)

            # Cosine similarity against all tracks
            sims = cosine_similarity(query_vec, emb_matrix).flatten()
            # Get top-K neighbors (exclude self)
            neighbor_indices = np.argsort(sims)[::-1][1:NEIGHBOR_K + 1]

            # Check agreement: how many neighbors have the old (wrong) value?
            agree_count = 0
            total_with_field = 0
            max_dist = 0.0

            for ni in neighbor_indices:
                ntid = track_ids[ni]
                nlabels = label_lookup.get(ntid, {})
                neighbor_val = self._get_field_value(nlabels, field)
                if neighbor_val is None:
                    continue
                total_with_field += 1
                if self._values_match(neighbor_val, old_value):
                    agree_count += 1
                    dist = 1.0 - sims[ni]
                    max_dist = max(max_dist, dist)

            if total_with_field == 0:
                continue

            agreement = agree_count / total_with_field

            if agreement >= AGREEMENT_THRESHOLD:
                # Recency weight: exponential decay
                created_at = corr.get("created_at")
                weight = self._recency_weight(created_at)
                confidence = agreement * weight

                pattern = CorrectionPattern(
                    center=emb_matrix[idx],
                    radius=max(max_dist, 0.05),  # minimum radius
                    field=field,
                    from_value=old_value,
                    to_value=new_value,
                    confidence=confidence,
                    correction_ids=[cid],
                )
                patterns.append(pattern)

        # Merge overlapping patterns
        patterns = self._merge_patterns(patterns)
        self._patterns = patterns

        logger.info(f"Found {len(patterns)} correction patterns from {len(corrections)} corrections")
        return patterns

    # ── Layer 3: Propagation ─────────────────────────────────────────────────

    async def propagate(self, dry_run: bool = False) -> dict:
        """
        Apply confident patterns to auto-tagged tracks.

        Only updates tracks with tag_source='auto' (never 'manual' or 'auto_reviewed').
        Returns a summary dict with counts.
        """
        if not self._patterns:
            await self.find_correction_patterns()

        if not self._patterns:
            return {"patterns": 0, "tracks_updated": 0, "dry_run": dry_run}

        # Filter to confident patterns only
        confident = [p for p in self._patterns if p.confidence >= CONFIDENCE_PROPAGATE]
        if not confident:
            return {
                "patterns": len(self._patterns),
                "confident_patterns": 0,
                "tracks_updated": 0,
                "dry_run": dry_run,
            }

        # Get auto-tagged tracks
        auto_tracks = await self._db.get_auto_tagged_tracks()
        if not auto_tracks:
            return {
                "patterns": len(self._patterns),
                "confident_patterns": len(confident),
                "tracks_updated": 0,
                "dry_run": dry_run,
            }

        # Build embedding matrix for auto tracks
        from backend.data.embeddings import build_embedding_matrix
        auto_ids, auto_matrix = build_embedding_matrix(auto_tracks)
        if len(auto_ids) == 0:
            return {
                "patterns": len(self._patterns),
                "confident_patterns": len(confident),
                "tracks_updated": 0,
                "dry_run": dry_run,
            }

        auto_id_to_idx = {tid: i for i, tid in enumerate(auto_ids)}
        label_lookup = await self._build_label_lookup(auto_ids)

        tracks_updated = 0
        corrections_applied = 0

        for pattern in confident:
            center_vec = pattern.center.reshape(1, -1)
            sims = cosine_similarity(center_vec, auto_matrix).flatten()
            distances = 1.0 - sims

            # Find tracks within the pattern radius
            in_radius = np.where(distances <= pattern.radius)[0]

            for idx in in_radius:
                tid = auto_ids[idx]
                labels = label_lookup.get(tid, {})
                current_val = self._get_field_value(labels, pattern.field)

                if current_val is None:
                    continue
                if not self._values_match(current_val, pattern.from_value):
                    continue

                # Apply correction
                if not dry_run:
                    new_val = json.loads(pattern.to_value) if isinstance(pattern.to_value, str) else pattern.to_value
                    updates = {pattern.field: new_val}
                    try:
                        updates["tag_source"] = "auto_reviewed"
                        await self._db.update_track_labels(tid, updates, log_corrections=False)
                    except Exception as e:
                        logger.error(f"Failed to propagate correction to {tid}: {e}")
                        continue

                tracks_updated += 1
                corrections_applied += 1

        # Log propagation
        if not dry_run and tracks_updated > 0:
            await self._db.log_propagation(corrections_applied, tracks_updated)
            self._last_propagation = datetime.utcnow()

        result = {
            "patterns": len(self._patterns),
            "confident_patterns": len(confident),
            "tracks_updated": tracks_updated,
            "corrections_applied": corrections_applied,
            "dry_run": dry_run,
        }
        logger.info(f"Propagation result: {result}")
        return result

    # ── Status ───────────────────────────────────────────────────────────────

    async def get_status(self) -> dict:
        """Return correction count, pattern count, and last propagation time."""
        corrections = await self._db.get_recent_corrections(since_days=HALF_LIFE_DAYS * 2)
        propagation_status = await self._db.get_propagation_status()

        return {
            "correction_count": len(corrections) if corrections else 0,
            "pattern_count": len(self._patterns),
            "last_propagation": propagation_status.get("last_propagation") if propagation_status else None,
            "total_propagated": propagation_status.get("total_propagated", 0) if propagation_status else 0,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    async def _fetch_tracks_with_embeddings(self) -> list[dict]:
        """Fetch all tracks that have embeddings from the database."""
        if not self._db.client:
            return []
        try:
            response = self._db.client.table("tracks") \
                .select("trackid, embedding") \
                .not_.is_("embedding", "null") \
                .execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to fetch tracks with embeddings: {e}")
            return []

    async def _build_label_lookup(self, track_ids: list[str]) -> dict[str, dict]:
        """Build a trackid → labels dict for a set of track IDs."""
        if not self._db.client or not track_ids:
            return {}
        try:
            # Supabase .in_() has a limit; batch if needed
            lookup = {}
            batch_size = 500
            for i in range(0, len(track_ids), batch_size):
                batch = track_ids[i:i + batch_size]
                response = self._db.client.table("track_labels") \
                    .select("trackid, semantic_tags, vibe, energy, tag_source") \
                    .in_("trackid", batch) \
                    .execute()
                for row in (response.data or []):
                    lookup[row["trackid"]] = row
            return lookup
        except Exception as e:
            logger.error(f"Failed to build label lookup: {e}")
            return {}

    @staticmethod
    def _get_field_value(labels: dict, field: str):
        """Extract a field value from a labels dict, parsing JSON strings."""
        val = labels.get(field)
        if val is None:
            return None
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val
        return val

    @staticmethod
    def _values_match(a, b) -> bool:
        """Compare two field values, handling JSON-encoded strings."""
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(b, str):
            try:
                b = json.loads(b)
            except (json.JSONDecodeError, TypeError):
                pass
        return a == b

    @staticmethod
    def _recency_weight(created_at) -> float:
        """Exponential decay weight with 30-day half-life."""
        if created_at is None:
            return 0.5
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return 0.5
        now = datetime.utcnow()
        if hasattr(created_at, 'replace'):
            created_at = created_at.replace(tzinfo=None)
        age_days = (now - created_at).total_seconds() / 86400
        return float(np.exp(-np.log(2) * age_days / HALF_LIFE_DAYS))

    @staticmethod
    def _merge_patterns(patterns: list[CorrectionPattern]) -> list[CorrectionPattern]:
        """
        Merge overlapping patterns that target the same field + from→to correction.
        Two patterns merge if their centers are within MERGE_COSINE_THRESHOLD similarity
        and they describe the same correction.
        """
        if len(patterns) <= 1:
            return patterns

        merged: list[CorrectionPattern] = []
        used = set()

        for i, pi in enumerate(patterns):
            if i in used:
                continue
            group_centers = [pi.center]
            group_ids = list(pi.correction_ids)
            group_radius = pi.radius
            group_confidence = pi.confidence

            for j, pj in enumerate(patterns):
                if j <= i or j in used:
                    continue
                if pj.field != pi.field or pj.from_value != pi.from_value or pj.to_value != pi.to_value:
                    continue
                sim = cosine_similarity(
                    pi.center.reshape(1, -1),
                    pj.center.reshape(1, -1),
                ).item()
                if sim >= MERGE_COSINE_THRESHOLD:
                    group_centers.append(pj.center)
                    group_ids.extend(pj.correction_ids)
                    group_radius = max(group_radius, pj.radius)
                    group_confidence = max(group_confidence, pj.confidence)
                    used.add(j)

            # Average center
            avg_center = np.mean(group_centers, axis=0).astype(np.float32)
            norm = np.linalg.norm(avg_center)
            if norm > 0:
                avg_center = avg_center / norm

            merged_pattern = CorrectionPattern(
                center=avg_center,
                radius=group_radius,
                field=pi.field,
                from_value=pi.from_value,
                to_value=pi.to_value,
                confidence=group_confidence,
                correction_ids=group_ids,
            )
            merged.append(merged_pattern)
            used.add(i)

        return merged
