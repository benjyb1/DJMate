"""
Semantic interpretation module for natural language DJ queries.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class SemanticInterpreter:
    """Converts natural language queries into structured parameters for DJ recommendations."""

    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the semantic interpreter.
        
        Args:
            model: OpenAI model to use for interpretation
        """
        self.model = model
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if self.openai_key:
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=self.openai_key)
                self.use_llm = True
            except ImportError:
                print("⚠️  OpenAI package not installed. Install with: pip install openai")
                self.use_llm = False
        else:
            print("⚠️  OPENAI_API_KEY not found in environment - using fallback interpretation")
            self.use_llm = False

        # DJ-specific semantic mappings for fallback
        self.semantic_schema = {
            "energy_descriptors": {
                "chill": (0.1, 0.3),
                "laid_back": (0.2, 0.4),
                "groovy": (0.4, 0.6),
                "driving": (0.6, 0.8),
                "intense": (0.7, 0.9),
                "banging": (0.8, 1.0)
            },
            "directional_terms": {
                "build": "build",
                "drop": "breakdown",
                "maintain": "maintain",
                "bridge": "bridge"
            }
        }

    async def interpret(
            self,
            query: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret natural language query and convert to structured format.
        
        Args:
            query: Natural language query from DJ
            context: Optional context from previous session
            
        Returns:
            Structured query parameters with confidence scores
        """
        if self.use_llm:
            return await self._llm_interpretation(query, context)
        else:
            return self._fallback_interpretation(query, context)

    async def _llm_interpretation(
            self,
            query: str,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM for intelligent interpretation."""

        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(query)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            parsed_result = json.loads(response.choices[0].message.content)

            # Validate and enhance
            return self._validate_and_enhance(parsed_result, context)

        except Exception as e:
            print(f"LLM interpretation error: {e}")
            return self._fallback_interpretation(query, context)

    def _build_system_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        """Build system prompt for LLM."""
        context_info = ""
        if context and context.get("current_track"):
            track = context["current_track"]
            context_info = f"""
Current track context:
- Title: {track.get('title', 'Unknown')}
- BPM: {track.get('bpm', 'Unknown')}
- Key: {track.get('key', 'Unknown')}
- Energy: {track.get('energy', 'Unknown')}
"""

        return f"""You are an expert DJ's assistant specialized in interpreting natural language requests about electronic music selection and mixing.

Your task is to convert DJ requests into structured JSON parameters for querying a music database.

{context_info}

Available parameters:
- tags: List of genre/style descriptors (e.g., ["deep house", "techno"])
- vibe_descriptors: List of mood/atmosphere terms (e.g., ["dark", "uplifting"])
- bpm_range: [min_bpm, max_bpm] or null
- energy_range: [min_energy, max_energy] (0.0-1.0 scale) or null
- key_compatibility: "same", "compatible", "any", or null
- direction: "build", "maintain", "breakdown", "bridge", or null
- confidence: Your confidence in this interpretation (0.0-1.0)
- notes: List of interpretation assumptions

Respond ONLY with valid JSON matching this schema. Be generous with tags - include related genres."""

    def _build_user_prompt(self, query: str) -> str:
        """Build user prompt."""
        return f'DJ Request: "{query}"\n\nConvert this to structured search parameters.'

    def _validate_and_enhance(
            self,
            parsed_result: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Post-process LLM output with DJ-specific logic."""

        # Ensure BPM range is mixable
        if "bpm_range" in parsed_result and parsed_result["bpm_range"]:
            bpm_min, bpm_max = parsed_result["bpm_range"]
            if bpm_max - bpm_min < 6:
                center = (bpm_min + bpm_max) / 2
                parsed_result["bpm_range"] = [center - 4, center + 4]

        # Add harmonic compatibility if context track provided
        if context and context.get("current_track") and "key_compatibility" not in parsed_result:
            parsed_result["key_compatibility"] = "compatible"

        # Ensure confidence
        if "confidence" not in parsed_result:
            parsed_result["confidence"] = 0.7

        return parsed_result

    def _fallback_interpretation(
            self,
            query: str,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable."""

        query_lower = query.lower()

        result = {
            "tags": [],
            "vibe_descriptors": [],
            "bpm_range": None,
            "energy_range": None,
            "direction": None,
            "confidence": 0.5,
            "notes": ["Fallback interpretation - LLM unavailable"]
        }

        # Extract energy descriptors
        for descriptor, (min_e, max_e) in self.semantic_schema["energy_descriptors"].items():
            if descriptor in query_lower:
                result["energy_range"] = [min_e, max_e]
                result["vibe_descriptors"].append(descriptor)

        # Extract direction
        for term, direction in self.semantic_schema["directional_terms"].items():
            if term in query_lower:
                result["direction"] = direction

        # Extract common genres
        genres = ["techno", "house", "trance", "drum and bass", "dubstep", "bass"]
        for genre in genres:
            if genre in query_lower:
                result["tags"].append(genre)

        return result
