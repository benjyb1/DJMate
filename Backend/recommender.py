"""
DJ recommendation engine with intelligent track selection logic.
Optimized for 2560-dimension embeddings without indexing.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

class DJRecommendationEngine:
    """Intelligent recommendation engine for DJ track selection."""

    def __init__(self, db_manager=None, embedding_index=None):
        self.db = db_manager
        # We handle similarity in-memory for datasets < 1000 songs
        self.camelot_compatibility = self._build_camelot_matrix()

    async def get_intelligent_recommendations(
            self,
            structured_query: Any,
            context_track_id: Optional[str] = None,
            apply_harmonic_weighting: bool = True,
            apply_energy_flow: bool = True,
            diversity_penalty: float = 0.2
    ) -> Dict[str, Any]:
        """Get intelligent track recommendations based on DJ logic."""
        if not self.db:
            return self._mock_recommendations()

        try:
            # 1. Resolve context track if provided
            context_track = None
            context_labels = {}
            if context_track_id:
                context_track = await self.db.get_track_by_id(context_track_id)
                context_labels = await self.db.get_track_labels(context_track_id)

            # 2. Convert query and fetch candidates
            query_dict = structured_query.dict() if hasattr(structured_query, 'dict') else structured_query
            candidates = await self.db.get_tracks_by_semantic_filter(query_dict)

            # 3. Comprehensive scoring
            scored_candidates = []
            for candidate in candidates:
                # Skip the context track itself
                if context_track and candidate.get("track_id") == context_track_id:
                    continue

                score_breakdown = self._compute_comprehensive_score(
                    candidate=candidate,
                    context_track=context_track,
                    context_labels=context_labels,
                    query=query_dict,
                    apply_harmonic_weighting=apply_harmonic_weighting,
                    apply_energy_flow=apply_energy_flow
                )

                scored_candidates.append({
                    "track": {k: v for k, v in candidate.items() if k != 'embedding'}, # Remove embedding from output
                    "total_score": score_breakdown["total"],
                    "score_breakdown": score_breakdown,
                    "reasoning": self._generate_reasoning(score_breakdown)
                })

            # 4. Sort and return
            scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)

            return {
                "tracks": scored_candidates[:20],
                "reasoning": {"method": "hybrid semantic + DJ logic scoring"},
                "overall_score": scored_candidates[0]["total_score"] if scored_candidates else 0.0
            }

        except Exception as e:
            print(f"Error in recommendations: {e}")
            return self._mock_recommendations()

    def _compute_comprehensive_score(
            self,
            candidate: Dict[str, Any],
            context_track: Optional[Dict[str, Any]],
            context_labels: Dict[str, Any],
            query: Dict[str, Any],
            apply_harmonic_weighting: bool,
            apply_energy_flow: bool
    ) -> Dict[str, float]:
        """Multi-factor scoring combining all DJ-relevant signals."""
        scores = {}

        # 1. Vector Similarity (2560 dimensions)
        if context_track and candidate.get("embedding") and context_track.get("embedding"):
            scores["embedding_similarity"] = self._calculate_cosine_similarity(
                candidate["embedding"],
                context_track["embedding"]
            )
        else:
            scores["embedding_similarity"] = 0.5

        # 2. BPM compatibility
        scores["bpm_compatibility"] = self._compute_bpm_compatibility(
            candidate.get("bpm"),
            query.get("bpm_range"),
            context_track.get("bpm") if context_track else None
        )

        # 3. Harmonic compatibility
        if apply_harmonic_weighting and context_track:
            scores["harmonic_compatibility"] = self._compute_harmonic_compatibility(
                candidate.get("key"),
                context_track.get("key")
            )
        else:
            scores["harmonic_compatibility"] = 0.5

        # 4. Energy flow
        candidate_energy = candidate.get("labels", {}).get("energy", 0.5)
        context_energy = context_labels.get("energy", 0.5)
        scores["energy_flow"] = self._compute_energy_flow_score(
            candidate_energy,
            context_energy,
            query.get("direction", "maintain")
        ) if apply_energy_flow else 0.5

        # Weights optimized for DJ transitions
        weights = {
            "embedding_similarity": 0.40,  # Higher weight for the 2560-dim "vibe"
            "bpm_compatibility": 0.20,
            "harmonic_compatibility": 0.25,
            "energy_flow": 0.15
        }

        total_score = sum(scores[f] * weights[f] for f in weights if f in scores)
        return {**scores, "total": float(total_score)}

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Fast cosine similarity using numpy for 2560-dim vectors."""
        try:
            v1, v2 = np.array(vec1), np.array(vec2)
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0: return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        except Exception:
            return 0.0

    def _compute_bpm_compatibility(self, cand_bpm, q_range, ctx_bpm) -> float:
        if not cand_bpm: return 0.5
        if q_range and (q_range[0] <= cand_bpm <= q_range[1]): return 1.0
        if ctx_bpm:
            diff = abs(cand_bpm - ctx_bpm)
            if diff <= 2: return 1.0
            if diff <= 5: return 0.8
            return max(0.1, 1.0 - (diff / 15))
        return 0.5

    def _compute_harmonic_compatibility(self, k1, k2) -> float:
        if not k1 or not k2: return 0.5
        c1, c2 = self._to_camelot(k1), self._to_camelot(k2)
        return self.camelot_compatibility.get((c1, c2), 0.2) if c1 and c2 else 0.5

    def _compute_energy_flow_score(self, c_en, ctx_en, direct) -> float:
        diff = c_en - ctx_en
        if direct == "build": return 1.0 if 0.05 <= diff <= 0.25 else 0.4 if diff > 0 else 0.1
        if direct == "breakdown": return 1.0 if -0.3 <= diff <= -0.1 else 0.4 if diff < 0 else 0.1
        return 1.0 if abs(diff) <= 0.15 else max(0.1, 1.0 - abs(diff) * 2)

    def _to_camelot(self, key: str) -> Optional[str]:
        """Basic musical key to Camelot notation mapping."""
        # Simple detection: If it ends in 'A' or 'B', it's already Camelot
        if key[-1] in ['A', 'B'] and key[:-1].isdigit(): return key

        mapping = {
            "C": "8B", "Am": "8A", "G": "9B", "Em": "9A", "D": "10B", "Bm": "10A",
            "A": "11B", "F#m": "11A", "E": "12B", "C#m": "12A", "B": "1B", "G#m": "1A",
            "F#": "2B", "D#m": "2A", "Db": "3B", "Bbm": "3A", "Ab": "4B", "Fm": "4A",
            "Eb": "5B", "Cm": "5A", "Bb": "6B", "Gm": "6A", "F": "7B", "Dm": "7A"
        }
        return mapping.get(key)

    def _build_camelot_matrix(self) -> Dict[Tuple[str, str], float]:
        """Builds a map of compatible Camelot keys."""
        matrix = {}
        for i in range(1, 13):
            for l in ['A', 'B']:
                k = f"{i}{l}"
                matrix[(k, k)] = 1.0 # Same key
                # Adjacent numbers
                prev_n = 12 if i == 1 else i - 1
                next_n = 1 if i == 12 else i + 1
                matrix[(k, f"{prev_n}{l}")] = 0.9
                matrix[(k, f"{next_n}{l}")] = 0.9
                # Relative Major/Minor
                other_l = 'B' if l == 'A' else 'A'
                matrix[(k, f"{i}{other_l}")] = 0.8
        return matrix

    def _generate_reasoning(self, scores: Dict[str, float]) -> str:
        if scores.get("total", 0) > 0.85: return "Perfect transition match"
        factors = []
        if scores.get("embedding_similarity", 0) > 0.8: factors.append("Similar vibe")
        if scores.get("harmonic_compatibility", 0) > 0.8: factors.append("Harmonic match")
        if scores.get("bpm_compatibility", 0) > 0.9: factors.append("Perfect BPM")
        return ", ".join(factors) if factors else "Good overall fit"

    def _mock_recommendations(self):
        return {"tracks": [], "reasoning": {"error": "DB unavailable"}}