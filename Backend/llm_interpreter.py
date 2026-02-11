import openai
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio
from dataclasses import dataclass

@dataclass
class InterpretationContext:
    current_track: Optional[Dict[str, Any]] = None
    recent_tracks: List[Dict[str, Any]] = None
    session_metadata: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None

class SemanticInterpreter:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = openai.AsyncOpenAI()

        # DJ-specific semantic mapping
        self.semantic_schema = {
            "energy_descriptors": {
                "chill": (0.1, 0.3), "laid_back": (0.2, 0.4), "groovy": (0.4, 0.6),
                "driving": (0.6, 0.8), "intense": (0.7, 0.9), "banging": (0.8, 1.0)
            },
            "directional_terms": {
                "build": "increase_energy", "drop": "decrease_energy",
                "maintain": "stable_energy", "bridge": "transitional"
            },
            "vibe_mappings": {
                "dark": ["dark", "moody", "underground"],
                "uplifting": ["uplifting", "euphoric", "positive"],
                "hypnotic": ["hypnotic", "repetitive", "trance-like"]
            }
        }

    async def interpret(self, natural_query: str, context: Optional[InterpretationContext] = None) -> Dict[str, Any]:
        """Convert natural language DJ request to structured parameters"""

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
                temperature=0.1  # Low temperature for consistent parsing
            )

            parsed_result = json.loads(response.choices[0].message.content)

            # Post-process and validate
            validated_result = self._validate_and_enhance(parsed_result, context)

            return validated_result

        except Exception as e:
            # Fallback to rule-based parsing
            return self._fallback_interpretation(natural_query, context)

    def _build_system_prompt(self, context: Optional[InterpretationContext]) -> str:
        context_info = ""
        if context and context.current_track:
            context_info = f"""
            Current track context:
            - Title: {context.current_track.get('title', 'Unknown')}
            - BPM: {context.current_track.get('bpm', 'Unknown')}
            - Key: {context.current_track.get('key', 'Unknown')}
            - Energy: {context.current_track.get('energy', 'Unknown')}
            - Tags: {context.current_track.get('tags', [])}
            """

        return f"""You are an expert DJ's assistant specialized in interpreting natural language requests about electronic music selection and mixing.

        Your task is to convert DJ requests into structured JSON parameters that can be used to query a music database.

        {context_info}

        Available parameters:
        - tags: List of genre/style descriptors
        - vibe_descriptors: List of mood/atmosphere terms
        - bpm_range: [min_bpm, max_bpm] or null
        - energy_range: [min_energy, max_energy] (0.0-1.0 scale)
        - key_compatibility: "same", "compatible", "any", or specific key
        - direction: "build_energy", "maintain_energy", "breakdown", "bridge"
        - exclude_tracks: List of track IDs to avoid
        - confidence: Your confidence in this interpretation (0.0-1.0)
        - notes: List of interpretation assumptions or clarifications

        Respond ONLY with valid JSON matching this schema."""

    def _build_user_prompt(self, query: str, context: Optional[InterpretationContext]) -> str:
        return f"""
        DJ Request: "{query}"
        
        Convert this to structured parameters for music search and recommendation.
        Consider DJ mixing requirements like harmonic compatibility and energy flow.
        """

    def _validate_and_enhance(self, parsed_result: Dict[str, Any], context: Optional[InterpretationContext]) -> Dict[str, Any]:
        """Post-process LLM output with DJ-specific logic"""

        # Enhance BPM ranges based on mixability
        if "bpm_range" in parsed_result and parsed_result["bpm_range"]:
            bpm_min, bpm_max = parsed_result["bpm_range"]
            # Ensure mixable range (typically ±8% for electronic music)
            if bmp_max - bpm_min < 6:
                center = (bpm_min + bpm_max) / 2
                parsed_result["bpm_range"] = [center - 4, center + 4]

        # Add harmonic compatibility logic
        if context and context.current_track and "key_compatibility" not in parsed_result:
            current_key = context.current_track.get("key")
            if current_key:
                parsed_result["key_compatibility"] = "compatible"
                parsed_result["reference_key"] = current_key

        # Ensure confidence score
        if "confidence" not in parsed_result:
            parsed_result["confidence"] = 0.7  # Default moderate confidence

        return parsed_result

    def _fallback_interpretation(self, query: str, context: Optional[InterpretationContext]) -> Dict[str, Any]:
        """Rule-based fallback when LLM fails"""
        # Simple keyword matching as backup
        result = {
            "tags": [],
            "vibe_descriptors": [],
            "confidence": 0.3,
            "notes": ["Fallback interpretation - LLM unavailable"]
        }

        # Basic keyword extraction
        query_lower = query.lower()
        for vibe, synonyms in self.semantic_schema["vibe_mappings"].items():
            if any(synonym in query_lower for synonym in synonyms):
                result["vibe_descriptors"].append(vibe)

        return result