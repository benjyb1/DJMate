import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from collections import defaultdict

@dataclass
class RecommendationContext:
    current_track_id: Optional[str] = None
    recent_track_ids: List[str] = None
    target_energy_direction: Optional[str] = None
    harmonic_preference: float = 0.7
    diversity_requirement: float = 0.3

class DJRecommendationEngine:
    def __init__(self, db_interface, embedding_index):
        self.db = db_interface
        self.embedding_index = embedding_index

        # Camelot Wheel relationships
        self.camelot_compatibility = self._build_camelot_matrix()

        # Energy flow preferences
        self.energy_flow_weights = {
            "build_energy": {"min_increase": 0.1, "max_increase": 0.3},
            "maintain_energy": {"tolerance": 0.15},
            "breakdown": {"min_decrease": 0.1, "max_decrease": 0.4},
            "bridge": {"tolerance": 0.2, "harmonic_weight": 0.9}
        }

    async def get_intelligent_recommendations(
            self,
            structured_query: Dict[str, Any],
            context_track_id: Optional[str] = None,
            apply_harmonic_weighting: bool = True,
            apply_energy_flow: bool = True,
            diversity_penalty: float = 0.2,
            limit: int = 20
    ) -> Dict[str, Any]:
        """Generate recommendations using multi-factor DJ logic"""

        # Get candidate tracks based on semantic filters
        candidates = await self._get_semantic_candidates(structured_query)

        # Get context information
        context_track = None
        if context_track_id:
            context_track = await self.db.get_track_by_id(context_track_id)

        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score_breakdown = await self._compute_comprehensive_score(
                candidate=candidate,
                context_track=context_track,
                query=structured_query,
                apply_harmonic_weighting=apply_harmonic_weighting,
                apply_energy_flow=apply_energy_flow
            )

            scored_candidates.append({
                "track": candidate,
                "total_score": score_breakdown["total"],
                "score_breakdown": score_breakdown,
                "reasoning": self._generate_reasoning(score_breakdown)
            })

        # Apply diversity penalty
        if diversity_penalty > 0:
            scored_candidates = self._apply_diversity_penalty(
                scored_candidates,
                penalty_weight=diversity_penalty
            )

        # Sort and limit results
        final_recommendations = sorted(
            scored_candidates,
            key=lambda x: x["total_score"],
            reverse=True
        )[:limit]

        # Generate pathway visualization data
        pathway_data = await self._generate_pathway_data(
            final_recommendations[:5],
            context_track
        )

        return {
            "recommendations": final_recommendations,
            "pathway_data": pathway_data,
            "query_interpretation": structured_query,
            "context_used": context_track_id is not None
        }

    async def _compute_comprehensive_score(
            self,
            candidate: Dict[str, Any],
            context_track: Optional[Dict[str, Any]],
            query: Dict[str, Any],
            apply_harmonic_weighting: bool,
            apply_energy_flow: bool
    ) -> Dict[str, float]:
        """Multi-factor scoring combining all DJ-relevant signals"""

        scores = {}

        # 1. Embedding similarity (if context track provided)
        if context_track:
            embedding_sim = await self._compute_embedding_similarity(
                candidate["embedding"],
                context_track["embedding"]
            )
            scores["embedding_similarity"] = embedding_sim
        else:
            scores["embedding_similarity"] = 0.5  # Neutral

        # 2. Semantic tag overlap
        scores["semantic_overlap"] = self._compute_semantic_overlap(
            candidate.get("tags", []),
            query.get("tags", [])
        )

        # 3. Vibe compatibility
        scores["vibe_compatibility"] = self._compute_vibe_compatibility(
            candidate.get("vibe_descriptors", []),
            query.get("vibe_descriptors", [])
        )

        # 4. BPM compatibility
        scores["bpm_compatibility"] = self._compute_bmp_compatibility(
            candidate.get("bpm"),
            query.get("bpm_range"),
            context_track.get("bpm") if context_track else None
        )

        # 5. Harmonic compatibility
        if apply_harmonic_weighting and context_track:
            scores["harmonic_compatibility"] = self._compute_harmonic_compatibility(
                candidate.get("key"),
                context_track.get("key")
            )
        else:
            scores["harmonic_compatibility"] = 0.5

        # 6. Energy flow compatibility
        if apply_energy_flow:
            scores["energy_flow"] = self._compute_energy_flow_score(
                candidate.get("energy", 0.5),
                context_track.get("energy", 0.5) if context_track else 0.5,
                query.get("direction", "maintain")
            )
        else:
            scores["energy_flow"] = 0.5

        # 7. Freshness/recency penalty
        scores["freshness"] = await self._compute_freshness_score(candidate["track_id"])

        # Weighted combination
        weights = {
            "embedding_similarity": 0.25,
            "semantic_overlap": 0.20,
            "vibe_compatibility": 0.15,
            "bpm_compatibility": 0.15,
            "harmonic_compatibility": 0.15 if apply_harmonic_weighting else 0.05,
            "energy_flow": 0.10 if apply_energy_flow else 0.05,
            "freshness": 0.05
        }

        total_score = sum(scores[factor] * weights[factor] for factor in scores)
        scores["total"] = total_score
        scores["weights_used"] = weights

        return scores

    def _compute_harmonic_compatibility(self, key1: Optional[str], key2: Optional[str]) -> float:
        """Camelot wheel-based harmonic compatibility"""
        if not key1 or not key2:
            return 0.5

        # Convert to Camelot notation if needed
        camelot1 = self._to_camelot(key1)
        camelot2 = self._to_camelot(key2)

        if not camelot1 or not camelot2:
            return 0.5

        return self.camelot_compatibility.get((camelot1, camelot2), 0.1)

    def _compute_energy_flow_score(self, candidate_energy: float, context_energy: float, direction: str) -> float:
        """Score based on desired energy progression"""
        energy_diff = candidate_energy - context_energy

        if direction == "build_energy":
            # Want positive energy increase
            if 0.1 <= energy_diff <= 0.3:
                return 1.0
            elif energy_diff > 0:
                return 0.7
            else:
                return 0.2

        elif direction == "breakdown":
            # Want negative energy decrease
            if -0.4 <= energy_diff <= -0.1:
                return 1.0
            elif energy_diff < 0:
                return 0.7
            else:
                return 0.2

        elif direction == "maintain":
            # Want stable energy
            if abs(energy_diff) <= 0.15:
                return 1.0
            else:
                return max(0.1, 1.0 - abs(energy_diff) * 2)

        elif direction == "bridge":
            # More tolerant, focus on harmonic compatibility
            if abs(energy_diff) <= 0.2:
                return 0.8
            else:
                return 0.5

        return 0.5

    async def validate_sequence(self, track_ids: List[str], sequence_order: List[int]) -> Dict[str, Any]:
        """Validate a sequence of tracks for mixing compatibility"""
        tracks = await self.db.get_tracks_by_ids(track_ids)
        ordered_tracks = [tracks[i] for i in sequence_order]

        issues = []
        overall_scores = []

        for i in range(len(ordered_tracks) - 1):
            current = ordered_tracks[i]
            next_track = ordered_tracks[i + 1]

            # Check BPM compatibility
            bpm_diff = abs(current.get("bpm", 120) - next_track.get("bpm", 120))
            if bpm_diff > 10:  # >10 BPM difference is challenging
                issues.append({
                    "type": "bpm_incompatibility",
                    "position": i,
                    "severity": "high" if bpm_diff > 15 else "medium",
                    "details": f"BPM jump: {current.get('bpm')} → {next_track.get('bpm')}"
                })

            # Check harmonic compatibility
            harmonic_score = self._compute_harmonic_compatibility(
                current.get("key"),
                next_track.get("key")
            )
            if harmonic_score < 0.3:
                issues.append({
                    "type": "harmonic_clash",
                    "position": i,
                    "severity": "medium",
                    "details": f"Key clash: {current.get('key')} → {next_track.get('key')}"
                })

            # Overall transition score
            transition_score = (
                    (1.0 - min(bpm_diff / 20, 1.0)) * 0.5 +  # BPM component
                    harmonic_score * 0.5  # Harmonic component
            )
            overall_scores.append(transition_score)

        return {
            "issues": issues,
            "transition_scores": overall_scores,
            "overall_score": np.mean(overall_scores) if overall_scores else 0.0,
            "is_mixable": len([issue for issue in issues if issue["severity"] == "high"]) == 0
        }

    def _build_camelot_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build Camelot wheel compatibility matrix"""
        # Simplified - in practice, you'd have a complete 24-key matrix
        compatibility = {}

        # Perfect matches
        for key in ["1A", "1B", "2A", "2B"]:  # etc.
            compatibility[(key, key)] = 1.0

        # Adjacent keys (±1 on wheel)
        adjacent_pairs = [
            ("1A", "2A"), ("1A", "12A"), ("1A", "1B"),
            # ... complete mapping
        ]

        for key1, key2 in adjacent_pairs:
            compatibility[(key1, key2)] = 0.8
            compatibility[(key2, key1)] = 0.8

        return compatibility