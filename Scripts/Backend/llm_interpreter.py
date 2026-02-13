from openai import AsyncOpenAI
import os
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from dotenv import load_dotenv
from supabase import Client

load_dotenv()


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
        # Energy descriptors mapping natural language to ranges
        self.energy_descriptors = {
            "low": (0.0, 0.3),
            "chill": (0.0, 0.3),
            "relaxed": (0.1, 0.4),
            "laid-back": (0.1, 0.4),
            "moderate": (0.3, 0.6),
            "medium": (0.3, 0.6),
            "groovy": (0.4, 0.7),
            "energetic": (0.5, 0.8),
            "high": (0.6, 0.9),
            "driving": (0.6, 0.9),
            "intense": (0.7, 1.0),
            "banging": (0.8, 1.0),
            "peak": (0.8, 1.0)
        }


class SemanticInterpreter:
    """
    LLM-powered interpreter for DJ requests that uses real database tags
    for accurate semantic matching and music selection.

    Uses Supabase client - no separate PostgreSQL connection needed.
    """

    def __init__(
            self,
            supabase_client: Client,
            model: str = "gemini-2.0-flash"
    ):
        self.model = model
        self.supabase = supabase_client

        # Initialize via Gemini's OpenAI-compatible endpoint (free tier: 15 RPM, 1M tokens/day)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        # Available tags (loaded from database)
        self.available_tags: Optional[AvailableTags] = None
        self._tags_loaded = False

    async def initialize(self):
        """Load available tags from database on startup"""
        await self._load_available_tags()

    async def _load_available_tags(self) -> AvailableTags:
        """Fetch all unique tags from track_labels table using Supabase"""

        try:
            # Get all track labels
            response = self.supabase.table("track_labels").select("semantic_tags, vibe").execute()

            tags = AvailableTags()

            for row in response.data:
                # Collect semantic tags
                semantic_tags = row.get('semantic_tags')
                if semantic_tags:
                    for tag in semantic_tags:
                        if isinstance(tag, str) and tag:
                            tags.semantic_tags.add(tag)
                        elif isinstance(tag, list):
                            for t in tag:
                                if isinstance(t, str) and t:
                                    tags.semantic_tags.add(t)

                # Collect vibes (single value per row)
                vibe = row.get('vibe')
                if vibe:
                    tags.vibes.add(vibe)

            self.available_tags = tags
            self._tags_loaded = True

            print(f"Loaded {len(tags.semantic_tags)} semantic tags and {len(tags.vibes)} vibes from database")
            return tags

        except Exception as e:
            print(f"Error loading tags from database: {e}")
            # Return empty tags as fallback
            self.available_tags = AvailableTags()
            return self.available_tags

    async def interpret(
            self,
            natural_query: str,
            context: Optional[InterpretationContext] = None
    ) -> Dict[str, Any]:
        """
        Convert natural language DJ request to structured parameters.

        Args:
            natural_query: User's natural language request (e.g., "play some dark techno")
            context: Current DJ session context

        Returns:
            Structured query parameters for backend
        """

        # Ensure tags are loaded
        if not self._tags_loaded:
            await self._load_available_tags()

        # Build prompts with real database tags
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(natural_query, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temp for consistent, precise parsing
            )

            parsed_result = json.loads(response.choices[0].message.content)

            # Validate and enhance with DJ-specific logic
            validated_result = await self._validate_and_enhance(parsed_result, context)

            return validated_result

        except Exception as e:
            print(f"LLM interpretation error: {e}")
            # Fallback to simpler rule-based parsing
            return await self._fallback_interpretation(natural_query, context)

    def _build_system_prompt(self, context: Optional[InterpretationContext]) -> str:
        """Build system prompt with available tags and context"""

        # Format available tags for LLM
        semantic_tags_list = sorted(list(self.available_tags.semantic_tags))
        vibes_list = sorted(list(self.available_tags.vibes))

        # Current track context
        context_info = ""
        if context and context.current_track:
            context_info = f"""
Current Track Context:
- Title: {context.current_track.get('title', 'Unknown')}
- BPM: {context.current_track.get('bpm', 'Unknown')}
- Key: {context.current_track.get('key', 'Unknown')}
- Energy: {context.current_track.get('energy', 'Unknown')}
- Semantic Tags: {context.current_track.get('semantic_tags', [])}
- Vibe: {context.current_track.get('vibe', 'Unknown')}
"""

        return f"""You are an expert DJ assistant that interprets natural language requests into structured database queries for music selection.

{context_info}

AVAILABLE TAGS IN DATABASE:

Semantic Tags (genres/styles): {json.dumps(semantic_tags_list)}

Vibes (moods/atmospheres): {json.dumps(vibes_list)}

Energy Descriptors: low/chill (0.0-0.3), moderate/medium (0.3-0.6), energetic/high (0.6-0.9), intense/peak (0.8-1.0)

YOUR TASK:
Parse the user's DJ request and match it to the ACTUAL tags available in the database above.
- Choose semantic_tags that best match the user's genre/style request
- Choose vibes that best match the user's mood/atmosphere request  
- Only set energy_range if the user explicitly mentions energy level (e.g., "high energy", "chill", "intense")
- Otherwise let energy naturally emerge from genre selection

OUTPUT SCHEMA (respond with ONLY valid JSON):
{{
    "semantic_tags": ["tag1", "tag2"],  // Must be from available semantic tags
    "vibes": ["vibe1"],  // Must be from available vibes, can be empty
    "energy_range": [min, max] or null,  // Only if explicitly mentioned, range 0.0-1.0
    "bpm_range": [min, max] or null,  // Only if BPM mentioned or needed for mixing
    "key_compatibility": "same" | "compatible" | "any" | null,  // For harmonic mixing
    "direction": "build" | "maintain" | "breakdown" | null,  // Energy trajectory
    "confidence": 0.0-1.0,  // Your confidence in this interpretation
    "reasoning": "Brief explanation of your choices",
    "suggestions": ["Alternative interpretations if ambiguous"]
}}

MATCHING RULES:
- Match user terms to the closest available tags (e.g., "techno" → find techno-related tags)
- If user says "dark techno", look for both "techno" in semantic_tags AND "dark" in vibes
- Be flexible with synonyms (e.g., "chill" could map to "ambient", "downtempo", etc.)
- If no exact match, choose the closest semantic equivalent
- Prefer specificity: "melodic techno" is better than just "techno"
- Consider DJ mixing context: harmonic compatibility, BPM ranges, energy flow"""

    def _build_user_prompt(self, query: str, context: Optional[InterpretationContext]) -> str:
        """Build user prompt with the query"""
        return f"""DJ Request: "{query}"

Parse this request and return structured parameters using ONLY the available tags from the database.
Think about what the DJ is trying to achieve in terms of genre, vibe, energy flow, and mixing requirements."""

    async def _validate_and_enhance(
            self,
            parsed_result: Dict[str, Any],
            context: Optional[InterpretationContext]
    ) -> Dict[str, Any]:
        """Post-process and validate LLM output with DJ-specific logic"""

        # Validate semantic tags against database
        if "semantic_tags" in parsed_result:
            valid_tags = [
                tag for tag in parsed_result["semantic_tags"]
                if tag in self.available_tags.semantic_tags
            ]
            parsed_result["semantic_tags"] = valid_tags

        # Validate vibes against database
        if "vibes" in parsed_result:
            valid_vibes = [
                vibe for vibe in parsed_result["vibes"]
                if vibe in self.available_tags.vibes
            ]
            parsed_result["vibes"] = valid_vibes

        # Enhance BPM ranges for mixability
        if parsed_result.get("bpm_range"):
            bpm_min, bpm_max = parsed_result["bpm_range"]
            # Ensure reasonable mixing range (±8% is standard for electronic music)
            if bpm_max - bpm_min < 6:
                center = (bpm_min + bpm_max) / 2
                parsed_result["bpm_range"] = [max(1, center - 4), center + 4]

        # Add harmonic compatibility if current track exists
        if context and context.current_track and not parsed_result.get("key_compatibility"):
            current_key = context.current_track.get("key")
            if current_key:
                parsed_result["key_compatibility"] = "compatible"
                parsed_result["reference_key"] = current_key

        # Ensure confidence score exists
        if "confidence" not in parsed_result:
            parsed_result["confidence"] = 0.7

        # Add interpretation metadata
        parsed_result["interpretation_method"] = "llm"
        parsed_result["model_used"] = self.model

        return parsed_result

    async def _fallback_interpretation(
            self,
            query: str,
            context: Optional[InterpretationContext]
    ) -> Dict[str, Any]:
        """Simple rule-based fallback when LLM fails"""

        result = {
            "semantic_tags": [],
            "vibes": [],
            "energy_range": None,
            "confidence": 0.3,
            "reasoning": "Fallback: LLM unavailable, using keyword matching",
            "interpretation_method": "fallback"
        }

        query_lower = query.lower()

        # Match semantic tags by keyword
        for tag in self.available_tags.semantic_tags:
            if tag.lower() in query_lower:
                result["semantic_tags"].append(tag)

        # Match vibes by keyword
        for vibe in self.available_tags.vibes:
            if vibe.lower() in query_lower:
                result["vibes"].append(vibe)

        # Detect energy descriptors
        for descriptor, energy_range in self.available_tags.energy_descriptors.items():
            if descriptor in query_lower:
                result["energy_range"] = list(energy_range)
                break

        return result

    async def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about tag usage in database"""

        try:
            # Get all track labels with their tags
            response = self.supabase.table("track_labels").select("semantic_tags").execute()

            # Count tag occurrences
            tag_counts = {}
            for row in response.data:
                tags = row.get('semantic_tags', [])
                if tags:
                    for tag in tags:
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Sort by count
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

            tag_stats = {
                "most_common_tags": [
                    {"tag": tag, "count": count}
                    for tag, count in sorted_tags[:20]
                ],
                "total_unique_tags": len(tag_counts)
            }

            return tag_stats

        except Exception as e:
            print(f"Error getting tag statistics: {e}")
            return {"most_common_tags": [], "total_unique_tags": 0}

    async def suggest_similar_tags(self, user_input: str, limit: int = 5) -> List[str]:
        """
        Suggest similar tags when user input doesn't match database exactly.
        Uses simple string similarity for now, could be enhanced with embeddings.
        """

        user_lower = user_input.lower()

        # Simple substring matching
        similar_semantic = [
            tag for tag in self.available_tags.semantic_tags
            if user_lower in tag.lower() or tag.lower() in user_lower
        ]

        similar_vibes = [
            vibe for vibe in self.available_tags.vibes
            if user_lower in vibe.lower() or vibe.lower() in user_lower
        ]

        return (similar_semantic + similar_vibes)[:limit]