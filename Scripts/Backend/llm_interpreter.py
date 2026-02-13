import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
from supabase import Client
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InterpretationContext:
    """Context about current DJ session and track"""
    current_track: Optional[Dict[str, Any]] = None
    recent_tracks: List[Dict[str, Any]] = field(default_factory=list)
    session_metadata: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None


@dataclass
class AvailableTags:
    """Cache of available tags from database"""
    semantic_tags: Set[str] = field(default_factory=set)
    vibes: Set[str] = field(default_factory=set)
    energy_descriptors: Dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        self.energy_descriptors = {
            "low": (0.0, 0.3), "chill": (0.0, 0.3), "relaxed": (0.1, 0.4),
            "laid-back": (0.1, 0.4), "moderate": (0.3, 0.6), "medium": (0.3, 0.6),
            "groovy": (0.4, 0.7), "energetic": (0.5, 0.8), "high": (0.6, 0.9),
            "driving": (0.6, 0.9), "intense": (0.7, 1.0), "banging": (0.8, 1.0),
            "peak": (0.8, 1.0)
        }


# ---------------------------------------------------------------------------
# Relaxation ladder — each step lowers the confidence threshold for
# which tags/vibes are included in the DB query.
# Scoring always uses the full confidence map; only the query widens.
# ---------------------------------------------------------------------------
_RELAXATION_STEPS = [
    {"tag_threshold": 0.65, "vibe_threshold": 0.65, "drop_ranges": False, "label": "high confidence"},
    {"tag_threshold": 0.45, "vibe_threshold": 0.45, "drop_ranges": False, "label": "medium confidence"},
    {"tag_threshold": 0.20, "vibe_threshold": 0.20, "drop_ranges": False, "label": "low confidence"},
    {"tag_threshold": 0.20, "vibe_threshold": 0.20, "drop_ranges": True,  "label": "no range filters"},
    {"tag_threshold": 0.0,  "vibe_threshold": 0.0,  "drop_ranges": True,  "label": "best effort"},
]


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    try:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


class SemanticInterpreter:
    """
    LLM-powered interpreter for DJ requests that uses real database tags.

    Features
    --------
    - Multi-provider fallback (Groq → Gemini → OpenAI → Mistral)
    - Sticky provider selection (remembers what works)
    - Supabase integration
    - Tag inference: extrapolates tags to untagged tracks via centroid embeddings
    - Progressive parameter relaxation: always returns something
    - LLM-controlled track count (no manual slider; parsed from natural language)
    """

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.available_tags: Optional[AvailableTags] = None
        self._tags_loaded = False
        self.providers: List[Dict[str, Any]] = []
        self._init_providers()
        self.active_provider_index = 0

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def _init_providers(self):
        if os.getenv("GROQ_API_KEY"):
            self.providers.append({
                "name": "Groq",
                "client": AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"),
                                      base_url="https://api.groq.com/openai/v1"),
                "model": "llama-3.3-70b-versatile"
            })
        if os.getenv("GEMINI_API_KEY"):
            self.providers.append({
                "name": "Gemini",
                "client": AsyncOpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
                "model": "gemini-2.0-flash"
            })
        if os.getenv("OPENAI_API_KEY"):
            self.providers.append({
                "name": "OpenAI",
                "client": AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                "model": "gpt-4o-mini"
            })
        if os.getenv("MISTRAL_API_KEY"):
            self.providers.append({
                "name": "Mistral",
                "client": AsyncOpenAI(api_key=os.getenv("MISTRAL_API_KEY"),
                                      base_url="https://api.mistral.ai/v1"),
                "model": "mistral-small-latest"
            })
        if not self.providers:
            logger.warning("⚠️  No LLM API keys found. Semantic interpretation will fail.")

    async def initialize(self):
        await self._load_available_tags()

    # -------------------------------------------------------------------------
    # Tag loading
    # -------------------------------------------------------------------------

    async def _load_available_tags(self) -> AvailableTags:
        tags = AvailableTags()
        try:
            tags_resp  = self.supabase.table("track_labels").select("semantic_tags").execute()
            vibes_resp = self.supabase.table("track_labels").select("vibe").execute()
            if tags_resp.data:
                for row in tags_resp.data:
                    if row.get("semantic_tags"):
                        tags.semantic_tags.update(row["semantic_tags"])
            if vibes_resp.data:
                for row in vibes_resp.data:
                    v = row.get("vibe")
                    if isinstance(v, list):
                        tags.vibes.update(v)
                    elif isinstance(v, str):
                        tags.vibes.add(v)
            logger.info(f"Loaded {len(tags.semantic_tags)} semantic tags, {len(tags.vibes)} vibes")
        except Exception as e:
            logger.error(f"Error loading tags: {e}")
        self.available_tags = tags
        self._tags_loaded = True
        return tags

    # -------------------------------------------------------------------------
    # Public: interpret a natural language query
    # -------------------------------------------------------------------------

    async def interpret(
            self,
            natural_query: str,
            context: Optional[InterpretationContext] = None,
    ) -> Dict[str, Any]:
        """
        Convert a natural language DJ request to structured search parameters.
        Returns a dict that includes `track_count` (LLM-parsed, default 5).
        """
        if not self._tags_loaded:
            await self._load_available_tags()
        if not self.providers:
            return await self._fallback_interpretation(natural_query, context)

        system_prompt = self._build_system_prompt(context)
        user_prompt   = self._build_user_prompt(natural_query)

        try:
            parsed   = await self._generate_with_fallback(system_prompt, user_prompt)
            validated = await self._validate_and_enhance(parsed, context)
            return validated
        except Exception as e:
            logger.error(f"All LLM providers failed: {e}")
            return await self._fallback_interpretation(natural_query, context)

    # -------------------------------------------------------------------------
    # Public: search with progressive relaxation + inference
    # -------------------------------------------------------------------------

    async def search(
            self,
            params: Dict[str, Any],
            db_manager=None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a track search using params from `interpret()`.

        Tries progressively looser constraints until `track_count` results are
        found. If the library is too sparse even at max relaxation, fills
        remaining slots with embedding-inferred tracks from untagged songs.
        """
        track_count = max(1, int(params.get("track_count", 5)))

        # Log what we're working with so quality issues are visible
        logger.info(
            f"🔎 Search params — tags: {params.get('semantic_tags')}, "
            f"vibes: {params.get('vibes')}, "
            f"energy: {params.get('energy_range')}, "
            f"bpm: {params.get('bpm_range')}, "
            f"count: {track_count}"
        )

        scored = []
        for step_idx, relaxation in enumerate(_RELAXATION_STEPS):
            relaxed = self._apply_relaxation(params, relaxation)
            tagged_tracks = await self._query_tagged_tracks(relaxed, track_count * 4)

            # IMPORTANT: always score against original params, not relaxed.
            # Relaxed params are only used to widen the DB query — once we
            # have candidate tracks we want to rank them by how well they
            # match what the user actually asked for.
            scored = self._score_tracks(tagged_tracks, params)[:track_count]

            logger.info(
                f"🔍 Step {step_idx} ({self._relaxation_label(step_idx)}): "
                f"fetched {len(tagged_tracks)}, scored {len(scored)} — "
                f"top score: {scored[0]['_relevance_score'] if scored else 'n/a'}"
            )

            if len(scored) >= track_count:
                return scored, {
                    "relaxation_step": step_idx,
                    "relaxation_label": self._relaxation_label(step_idx),
                    "inferred_count": 0,
                    "total_found": len(scored),
                }

        # After all relaxation steps, fill gaps with inferred (untagged) tracks
        best_tagged  = scored
        still_needed = track_count - len(best_tagged)
        found_ids    = {str(t.get("trackid") or "") for t in best_tagged}

        inferred: List[Dict[str, Any]] = []
        if still_needed > 0 and db_manager is not None:
            inferred = await self._infer_from_embeddings(
                params=params,
                db_manager=db_manager,
                exclude_ids=found_ids,
                needed=still_needed,
            )
        elif still_needed > 0:
            inferred = await self._infer_from_audio_features(
                params=params,
                exclude_ids=found_ids,
                needed=still_needed,
            )

        all_tracks = best_tagged + inferred
        return all_tracks, {
            "relaxation_step": len(_RELAXATION_STEPS) - 1,
            "relaxation_label": "best effort (full relaxation + inference)",
            "inferred_count": len(inferred),
            "total_found": len(all_tracks),
        }

    # -------------------------------------------------------------------------
    # Relaxation helpers
    # -------------------------------------------------------------------------

    def _apply_relaxation(self, params: Dict[str, Any], relaxation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of params with the query-side tag/vibe lists filtered
        to only those above the current confidence threshold.
        Scoring always uses the full tag_scores/vibe_scores maps.
        """
        p = dict(params)

        tag_threshold  = relaxation["tag_threshold"]
        vibe_threshold = relaxation["vibe_threshold"]

        tag_scores  = params.get("tag_scores")  or {}
        vibe_scores = params.get("vibe_scores") or {}

        # Tags/vibes to actually query at this relaxation level
        p["_query_tags"]  = [t for t, s in tag_scores.items()  if s >= tag_threshold]
        p["_query_vibes"] = [v for v, s in vibe_scores.items() if s >= vibe_threshold]

        if relaxation.get("drop_ranges"):
            p["energy_range"] = None
            p["bpm_range"]    = None

        return p

    def _relaxation_label(self, step: int) -> str:
        return _RELAXATION_STEPS[step]["label"] if step < len(_RELAXATION_STEPS) else "best effort"

    # -------------------------------------------------------------------------
    # Database query (tagged tracks)
    # -------------------------------------------------------------------------

    async def _query_tagged_tracks(
            self,
            params: Dict[str, Any],
            limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Two-step join:
          1. Query track_labels (trackid, semantic_tags, vibe, energy)
          2. Fetch matching tracks (trackid, bpm, key, title, artist, filepath, embedding)

        Supabase/PostgREST jsonb arrays don't support .overlaps() — that
        operator works on text[] only. For jsonb we use the @> (contains)
        operator via .filter("col", "cs", '[\"val\"]'), chained as OR via
        .or_() for multiple values.
        """
        try:
            query_tags  = params.get("_query_tags",  params.get("semantic_tags") or [])
            query_vibes = params.get("_query_vibes", params.get("vibes") or [])

            # ── Step 1: filter track_labels ───────────────────────────────────
            labels_q = self.supabase.table("track_labels").select(
                "trackid, semantic_tags, vibe, energy"
            )

            # jsonb "contains at least one of these values" via OR of cs filters.
            # cs.[\"X\"] means the jsonb array contains the element X.
            if query_tags:
                or_parts = ",".join(
                    f'semantic_tags.cs.{json.dumps([t])}' for t in query_tags
                )
                labels_q = labels_q.or_(or_parts)

            if query_vibes:
                or_parts = ",".join(
                    f'vibe.cs.{json.dumps([v])}' for v in query_vibes
                )
                labels_q = labels_q.or_(or_parts)

            if params.get("energy_range"):
                lo, hi = params["energy_range"]
                labels_q = labels_q.gte("energy", lo).lte("energy", hi)

            labels_resp = labels_q.limit(limit * 3).execute()
            label_rows  = labels_resp.data or []

            if not label_rows:
                return []

            # ── Step 2: fetch track metadata for matched IDs ──────────────────
            ids = [r["trackid"] for r in label_rows if r.get("trackid")]
            tracks_resp = (
                self.supabase.table("tracks")
                .select("trackid, title, artist, bpm, key, filepath, embedding")
                .in_("trackid", ids)
                .execute()
            )
            tracks_by_id = {r["trackid"]: r for r in (tracks_resp.data or [])}

            # ── Step 3: merge and apply bpm filter ────────────────────────────
            merged = []
            bpm_range = params.get("bpm_range")
            for label in label_rows:
                tid   = label.get("trackid")
                track = tracks_by_id.get(tid, {})

                bpm = track.get("bpm")
                if bpm_range and bpm is not None:
                    if not (bpm_range[0] <= float(bpm) <= bpm_range[1]):
                        continue

                merged.append({
                    "trackid"      : tid,
                    "semantic_tags": label.get("semantic_tags") or [],
                    "vibe"         : label.get("vibe") or [],
                    "energy"       : label.get("energy"),
                    "bpm"          : bpm,
                    "key"          : track.get("key"),
                    "title"        : track.get("title"),
                    "artist"       : track.get("artist"),
                    "filepath"     : track.get("filepath"),
                    "embedding"    : track.get("embedding"),
                })
                if len(merged) >= limit:
                    break

            return merged

        except Exception as e:
            logger.error(f"DB query error: {e}")
            return []

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _score_tracks(
            self,
            tracks: List[Dict[str, Any]],
            params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Score each track using confidence-weighted tag/vibe overlap.

        For each tag a track has, look up the LLM's confidence score for that
        tag and add it proportionally. This means:
          - A track tagged "house" scores 0.9 if the LLM was 90% confident
            about house
          - A track tagged only "minimal techno" scores 0.3 if that was a
            low-confidence suggestion
          - Tracks matching multiple confident tags score highest

        Energy and BPM contribute additional signal when present.
        """
        tag_scores   = params.get("tag_scores")   or {}
        vibe_scores  = params.get("vibe_scores")  or {}
        energy_range = params.get("energy_range")
        bpm_range    = params.get("bpm_range")

        # Max possible tag/vibe score (for normalisation)
        max_tag_score  = sum(tag_scores.values())  or 1.0
        max_vibe_score = sum(vibe_scores.values()) or 1.0

        # Lowercase lookup for case-insensitive matching at score time
        tag_scores_lower  = {k.lower(): v for k, v in tag_scores.items()}
        vibe_scores_lower = {k.lower(): v for k, v in vibe_scores.items()}

        has_energy = bool(energy_range)
        has_bpm    = bool(bpm_range)
        has_tags   = bool(tag_scores)
        has_vibes  = bool(vibe_scores)

        scored = []
        for t in tracks:
            score = 0.0

            # ── Tag score (0–0.45) ────────────────────────────────────────────
            if has_tags:
                track_tags = [tg.lower() for tg in (t.get("semantic_tags") or [])]
                tag_hit    = sum(tag_scores_lower.get(tg, 0.0) for tg in track_tags)
                score     += (tag_hit / max_tag_score) * 0.45

            # ── Vibe score (0–0.35) ───────────────────────────────────────────
            if has_vibes:
                tv = t.get("vibe") or []
                track_vibes = [v.lower() for v in (tv if isinstance(tv, list) else [tv])]
                vibe_hit    = sum(vibe_scores_lower.get(v, 0.0) for v in track_vibes)
                score      += (vibe_hit / max_vibe_score) * 0.35

            # ── Energy proximity (0–0.12) ─────────────────────────────────────
            if has_energy and t.get("energy") is not None:
                lo, hi = energy_range
                mid    = (lo + hi) / 2
                span   = max(hi - lo, 0.1)
                dist   = abs(float(t["energy"]) - mid) / span
                score += max(0.0, 1.0 - dist) * 0.12

            # ── BPM proximity (0–0.08) ────────────────────────────────────────
            if has_bpm and t.get("bpm") is not None:
                lo, hi = bpm_range
                mid    = (lo + hi) / 2
                span   = max(hi - lo, 10)
                dist   = abs(float(t["bpm"]) - mid) / span
                score += max(0.0, 1.0 - dist) * 0.08

            # ── Zero-signal fallback ──────────────────────────────────────────
            if not has_tags and not has_vibes and not has_energy and not has_bpm:
                score = 0.5

            t = dict(t)
            t.setdefault("trackid", t.pop("track_id", None))
            t["_relevance_score"] = round(min(score, 1.0), 3)

            # Attach per-track score breakdown for the UI to optionally display
            t["_score_detail"] = {
                "matched_tags": {
                    tg: round(tag_scores_lower.get(tg.lower(), 0.0), 2)
                    for tg in (t.get("semantic_tags") or [])
                    if tg.lower() in tag_scores_lower
                },
                "matched_vibes": {
                    v: round(vibe_scores_lower.get(v.lower(), 0.0), 2)
                    for v in (t.get("vibe") or [])
                    if v.lower() in vibe_scores_lower
                },
            }
            scored.append(t)

        scored.sort(key=lambda x: x["_relevance_score"], reverse=True)
        return scored

    # -------------------------------------------------------------------------
    # ★ Inference path A: use embeddings via DatabaseManager (preferred)
    # -------------------------------------------------------------------------

    async def _infer_from_embeddings(
            self,
            params: Dict[str, Any],
            db_manager,  # DatabaseManager from streamSimilar
            exclude_ids: Set[str],
            needed: int,
    ) -> List[Dict[str, Any]]:
        """
        Build a centroid embedding from the tagged tracks that match the
        request, then use db_manager.find_similar_tracks() — the same vector
        search powering streamSimilar — to find untagged tracks nearby.

        This means "upbeat deep house" will find tracks that *sound like*
        upbeat deep house even if they've never been labelled.
        """
        try:
            # 1. Fetch tagged reference tracks to build centroid
            reference_tracks = await self._query_tagged_tracks(params, limit=30)
            embeddings = [
                t["embedding"] for t in reference_tracks
                if t.get("embedding") and str(t.get("trackid") or "") not in exclude_ids
            ]

            if not embeddings:
                logger.info("🔮 No reference embeddings found, falling back to audio-feature inference")
                return await self._infer_from_audio_features(params, exclude_ids, needed)

            # 2. Average embeddings → centroid
            centroid = np.mean([np.array(e, dtype=np.float32) for e in embeddings], axis=0).tolist()
            logger.info(f"🔮 Built centroid from {len(embeddings)} reference tracks")

            # 3. Find all tracks near the centroid
            raw_similar = await db_manager.find_similar_tracks(
                query_embedding=centroid,
                limit=needed * 5,    # fetch extra so we can filter excludes
                threshold=0.25,      # fairly permissive — we score below
            )

            # 4. Filter out already-found tracks and enrich
            inferred = []
            for r in raw_similar:
                rid = str(r.get("id") or r.get("trackid") or "")
                if rid in exclude_ids:
                    continue

                full = await db_manager.get_track_by_id(rid)
                if full is None:
                    continue

                # Re-compute precise cosine similarity against centroid
                c_emb = getattr(full, "embedding", None)
                sim   = _cosine_similarity(centroid, c_emb) if c_emb else float(r.get("similarity", 0.3))

                track_dict = self._track_obj_to_dict(full)
                track_dict["_relevance_score"] = round(sim, 3)
                track_dict["_inferred"]        = True
                track_dict["_inferred_reason"] = "embedding similarity to tagged references"
                inferred.append(track_dict)

                if len(inferred) >= needed:
                    break

            inferred.sort(key=lambda x: x["_relevance_score"], reverse=True)
            logger.info(f"🔮 Inferred {len(inferred)} track(s) via embeddings")
            return inferred[:needed]

        except Exception as e:
            logger.error(f"Embedding inference error: {e}")
            return await self._infer_from_audio_features(params, exclude_ids, needed)

    # -------------------------------------------------------------------------
    # ★ Inference path B: audio-feature heuristic (no embeddings needed)
    # -------------------------------------------------------------------------

    async def _infer_from_audio_features(
            self,
            params: Dict[str, Any],
            exclude_ids: Set[str],
            needed: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback inference when no db_manager is available.
        Scores untagged tracks by BPM proximity (from tracks table) and
        energy proximity (from track_labels table, joined by trackid).
        """
        try:
            target_energy = None
            if params.get("energy_range"):
                lo, hi = params["energy_range"]
                target_energy = (lo + hi) / 2

            target_bpm = None
            if params.get("bpm_range"):
                lo, hi = params["bpm_range"]
                target_bpm = (lo + hi) / 2

            # Fetch all tracks (bpm lives here)
            tracks_resp = (
                self.supabase.table("tracks")
                .select("trackid, title, artist, bpm, key, filepath, embedding")
                .limit(500)
                .execute()
            )
            all_tracks = {r["trackid"]: r for r in (tracks_resp.data or [])}

            # Fetch energy for all track_labels (energy lives here)
            energy_resp = (
                self.supabase.table("track_labels")
                .select("trackid, energy")
                .execute()
            )
            energy_by_id = {r["trackid"]: r.get("energy") for r in (energy_resp.data or [])}

            candidates = []
            for tid, t in all_tracks.items():
                if tid in exclude_ids:
                    continue

                score = 0.5  # neutral baseline

                energy = energy_by_id.get(tid)
                if target_energy is not None and energy is not None:
                    dist   = abs(float(energy) - target_energy)
                    score += max(0.0, 0.5 - dist)

                if target_bpm is not None and t.get("bpm") is not None:
                    dist   = abs(float(t["bpm"]) - target_bpm) / 30.0
                    score += max(0.0, 0.5 - dist)

                candidates.append({
                    "trackid"       : tid,
                    "title"         : t.get("title"),
                    "artist"        : t.get("artist"),
                    "bpm"           : t.get("bpm"),
                    "key"           : t.get("key"),
                    "energy"        : energy,
                    "filepath"      : t.get("filepath"),
                    "embedding"     : t.get("embedding"),
                    "semantic_tags" : [],
                    "vibe"          : [],
                    "_relevance_score": round(min(score, 1.0), 3),
                    "_inferred"     : True,
                    "_inferred_reason": "audio feature proximity (untagged)",
                })

            candidates.sort(key=lambda x: x["_relevance_score"], reverse=True)
            result = candidates[:needed]
            logger.info(f"🔮 Inferred {len(result)} track(s) via audio features")
            return result

        except Exception as e:
            logger.error(f"Audio feature inference error: {e}")
            return []

    # -------------------------------------------------------------------------
    # LLM generation
    # -------------------------------------------------------------------------

    async def _generate_with_fallback(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        attempts    = 0
        current_idx = self.active_provider_index

        while attempts < len(self.providers):
            provider = self.providers[current_idx]
            try:
                response = await provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=10.0,
                )
                content = response.choices[0].message.content
                parsed  = json.loads(content)

                if current_idx != self.active_provider_index:
                    logger.info(f"✅ Switched active provider to {provider['name']}")
                    self.active_provider_index = current_idx

                parsed["model_used"] = f"{provider['name']} ({provider['model']})"
                return parsed

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                logger.warning(f"⚠️  {provider['name']} unavailable ({type(e).__name__}), switching…")
            except Exception as e:
                logger.warning(f"⚠️  {provider['name']} error: {e}, switching…")

            current_idx = (current_idx + 1) % len(self.providers)
            attempts   += 1

        raise RuntimeError("All configured LLM providers failed.")

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------

    def _build_system_prompt(self, context: Optional[InterpretationContext]) -> str:
        semantic_tags_list = sorted(self.available_tags.semantic_tags)
        vibes_list         = sorted(self.available_tags.vibes)

        context_info = ""
        if context and context.current_track:
            t = context.current_track
            context_info = (
                f"\nCurrent Track: {t.get('title','Unknown')} — "
                f"Key: {t.get('key','?')}  BPM: {t.get('bpm','?')}\n"
            )

        return f"""You are an expert DJ assistant interpreting messy natural language requests.
{context_info}
AVAILABLE DATABASE TAGS:
Genres/Styles : {json.dumps(semantic_tags_list)}
Vibes         : {json.dumps(vibes_list)}

TASK:
The user will describe what they want to hear in any words they like.
Your job is to map their intent to the tags above using CONFIDENCE SCORES (0.0–1.0).

Rules:
- Only use tags from the lists above — no invented tags
- Score reflects how well the tag captures the user's intent
- Include tags the user didn't name but that fit the vibe (with lower scores)
- A messy query like "fast kicking house" might map:
    tag_scores:  {{"house": 0.9, "tech house": 0.75, "minimal techno": 0.3}}
    vibe_scores: {{"energetic": 0.9, "driving": 0.8, "dark": 0.2}}
- "give me 6 tracks" → track_count: 6  |  no count mentioned → track_count: 5

OUTPUT — valid JSON only, no markdown:
{{
    "tag_scores"       : {{"tag_name": 0.0-1.0}},
    "vibe_scores"      : {{"vibe_name": 0.0-1.0}},
    "energy_range"     : [min, max] or null,
    "bpm_range"        : [min, max] or null,
    "key_compatibility": "same" | "compatible" | "any" | null,
    "direction"        : "build" | "maintain" | "breakdown" | null,
    "track_count"      : 5,
    "confidence"       : 0.0-1.0,
    "reasoning"        : "what you understood from the request",
    "suggestions"      : ["alternative phrasing if unsure"]
}}
"""

    def _build_user_prompt(self, query: str) -> str:
        return f'DJ Request: "{query}"'

    # -------------------------------------------------------------------------
    # Validate & enhance LLM output
    # -------------------------------------------------------------------------

    async def _validate_and_enhance(
            self,
            parsed: Dict[str, Any],
            context: Optional[InterpretationContext],
    ) -> Dict[str, Any]:
        """
        Validate LLM output, normalise tag/vibe case against DB, and emit
        both scored maps (tag_scores, vibe_scores) and flat lists
        (semantic_tags, vibes) for backwards compatibility.
        """
        tag_lower_map  = {t.lower(): t for t in self.available_tags.semantic_tags}
        vibe_lower_map = {v.lower(): v for v in self.available_tags.vibes}

        # ── Normalise tag_scores ──────────────────────────────────────────────
        raw_tag_scores = parsed.get("tag_scores") or {}
        # Handle old-style flat list from fallback/old prompts gracefully
        if not raw_tag_scores and parsed.get("semantic_tags"):
            raw_tag_scores = {t: 1.0 for t in parsed["semantic_tags"]}

        tag_scores: Dict[str, float] = {}
        stripped_tags = []
        for tag, score in raw_tag_scores.items():
            canonical = tag_lower_map.get(tag.lower())
            if canonical:
                tag_scores[canonical] = round(float(score), 3)
            else:
                stripped_tags.append(tag)
        if stripped_tags:
            logger.warning(
                f"⚠️  Tags not in DB (stripped): {stripped_tags}\n"
                f"    DB has: {sorted(tag_lower_map.keys())}"
            )

        # ── Normalise vibe_scores ─────────────────────────────────────────────
        raw_vibe_scores = parsed.get("vibe_scores") or {}
        if not raw_vibe_scores and parsed.get("vibes"):
            raw_vibe_scores = {v: 1.0 for v in parsed["vibes"]}

        vibe_scores: Dict[str, float] = {}
        stripped_vibes = []
        for vibe, score in raw_vibe_scores.items():
            canonical = vibe_lower_map.get(vibe.lower())
            if canonical:
                vibe_scores[canonical] = round(float(score), 3)
            else:
                stripped_vibes.append(vibe)
        if stripped_vibes:
            logger.warning(
                f"⚠️  Vibes not in DB (stripped): {stripped_vibes}\n"
                f"    DB has: {sorted(vibe_lower_map.keys())}"
            )

        parsed["tag_scores"]  = tag_scores   # e.g. {"house": 0.9, "tech house": 0.7}
        parsed["vibe_scores"] = vibe_scores  # e.g. {"energetic": 0.9, "driving": 0.8}

        # Flat lists (all tags with any score > 0) for backwards compat
        parsed["semantic_tags"] = list(tag_scores.keys())
        parsed["vibes"]         = list(vibe_scores.keys())

        # ── BPM range ─────────────────────────────────────────────────────────
        if parsed.get("bpm_range"):
            lo, hi = parsed["bpm_range"]
            if hi - lo < 6:
                mid = (lo + hi) / 2
                parsed["bpm_range"] = [max(1, mid - 4), mid + 4]

        # ── Key compatibility from context ────────────────────────────────────
        if context and context.current_track and not parsed.get("key_compatibility"):
            if context.current_track.get("key"):
                parsed["key_compatibility"] = "compatible"
                parsed["reference_key"]     = context.current_track["key"]

        parsed.setdefault("confidence",  0.7)
        parsed.setdefault("track_count", 5)
        parsed["track_count"] = max(1, int(parsed["track_count"]))

        logger.info(
            f"✅ Interpreted — tag_scores: {tag_scores}, "
            f"vibe_scores: {vibe_scores}, "
            f"energy: {parsed.get('energy_range')}, "
            f"bpm: {parsed.get('bpm_range')}"
        )

        parsed["interpretation_method"] = "llm"
        return parsed

    # -------------------------------------------------------------------------
    # Fallback (no LLM)
    # -------------------------------------------------------------------------

    async def _fallback_interpretation(
            self,
            query: str,
            context: Optional[InterpretationContext],
    ) -> Dict[str, Any]:
        logger.warning("Using keyword-matching fallback interpretation.")
        q = query.lower()

        # Simple keyword match — score 1.0 for direct hits
        tag_scores  = {t: 1.0 for t in self.available_tags.semantic_tags if t.lower() in q}
        vibe_scores = {v: 1.0 for v in self.available_tags.vibes          if v.lower() in q}

        energy_range = None
        for desc, rng in self.available_tags.energy_descriptors.items():
            if desc in q:
                energy_range = list(rng)
                break

        import re
        m = re.search(r'\b(\d+)\s*(?:track|song|result|tune)s?\b', q)
        track_count = max(1, int(m.group(1))) if m else 5

        return {
            "tag_scores"           : tag_scores,
            "vibe_scores"          : vibe_scores,
            "semantic_tags"        : list(tag_scores.keys()),
            "vibes"                : list(vibe_scores.keys()),
            "energy_range"         : energy_range,
            "bpm_range"            : None,
            "track_count"          : track_count,
            "confidence"           : 0.3,
            "reasoning"            : "Fallback: keyword matching",
            "interpretation_method": "fallback",
        }

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _track_obj_to_dict(self, track_obj) -> Dict[str, Any]:
        """
        Convert a track ORM/dataclass object (from DatabaseManager) to a plain dict.
        Note: energy lives in track_labels, not on the tracks object — it will be
        None here unless the ORM has already joined it in.
        """
        if isinstance(track_obj, dict):
            d = dict(track_obj)
        else:
            d = {
                "trackid"      : str(getattr(track_obj, "trackid",       None) or ""),
                "title"        : getattr(track_obj, "title",              None),
                "artist"       : getattr(track_obj, "artist",             None),
                "bpm"          : getattr(track_obj, "bpm",                None),
                "key"          : getattr(track_obj, "key",                None),
                "energy"       : getattr(track_obj, "energy",             None),  # may be None
                "semantic_tags": getattr(track_obj, "semantic_tags",      None) or [],
                "vibe"         : getattr(track_obj, "vibe_descriptors",   None) or [],
                "filepath"     : getattr(track_obj, "filepath",           None),
                "embedding"    : getattr(track_obj, "embedding",          None),
            }
        d.setdefault("trackid", str(d.pop("track_id", "") or ""))
        return d

    # -------------------------------------------------------------------------
    # Utility queries (unchanged from original)
    # -------------------------------------------------------------------------

    async def get_tag_statistics(self) -> Dict[str, Any]:
        try:
            resp = self.supabase.table("track_labels").select("semantic_tags").execute()
            counts: Dict[str, int] = {}
            for row in resp.data:
                for tag in (row.get("semantic_tags") or []):
                    if tag:
                        counts[tag] = counts.get(tag, 0) + 1
            sorted_tags = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return {
                "most_common_tags": [{"tag": t, "count": c} for t, c in sorted_tags[:20]],
                "total_unique_tags": len(counts),
            }
        except Exception as e:
            logger.error(f"Error getting tag statistics: {e}")
            return {"most_common_tags": [], "total_unique_tags": 0}

    async def suggest_similar_tags(self, user_input: str, limit: int = 5) -> List[str]:
        q = user_input.lower()
        return (
                       [t for t in self.available_tags.semantic_tags if q in t.lower() or t.lower() in q]
                       + [v for v in self.available_tags.vibes        if q in v.lower() or v.lower() in q]
               )[:limit]