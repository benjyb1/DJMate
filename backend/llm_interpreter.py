import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from supabase import Client
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

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
# Tool definitions for LLM function-calling.
# The LLM decides which tools to call based on the query — it never has to
# fill in a fixed JSON schema, so optional fields are truly optional.
# ---------------------------------------------------------------------------
_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_genre",
            "description": (
                "Call this when the user names a music genre or style. "
                "Use the primary genre with high confidence (0.85-1.0). "
                "Also include closely related/neighbouring genres at lower confidence "
                "(0.3-0.65) so the search can widen naturally when not enough results are found. "
                "Genre names MUST match entries from the Genres/Styles list exactly. "
                "Do NOT put mood/vibe words here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":       {"type": "string", "description": "Exact genre name from the available list"},
                                "confidence": {"type": "number", "description": "0.85-1.0 for the named genre; 0.3-0.65 for related genres"},
                            },
                            "required": ["name", "confidence"],
                        },
                    }
                },
                "required": ["genres"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vibe",
            "description": (
                "Call this when the user uses adjectives, moods, or atmosphere words. "
                "Every descriptor that is NOT a genre name goes here: dark, warm, hypnotic, "
                "bouncy, euphoric, groovy, driving, aggressive, melancholic, etc. "
                "Use the named vibe with high confidence (0.85-1.0). "
                "Also include closely related vibes at lower confidence (0.3-0.65) for widening. "
                "Vibe names MUST match entries from the Vibes list exactly. "
                "Do NOT put genre names here. NEVER convert a vibe into a genre."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vibes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":       {"type": "string", "description": "Exact vibe name from the available list"},
                                "confidence": {"type": "number", "description": "0.85-1.0 for the named vibe; 0.3-0.65 for related vibes"},
                            },
                            "required": ["name", "confidence"],
                        },
                    }
                },
                "required": ["vibes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_energy",
            "description": (
                "Call this when you can infer an energy level from the user's request. "
                "Energy is on a 1-10 integer scale. Common mappings: "
                "'absolute banger'/'peak time'/'kicking' -> [9, 10]; "
                "'high energy'/'banging' -> [8, 10]; "
                "'energetic'/'driving' -> [6, 8]; "
                "'mid-set'/'groovy' -> [5, 7]; "
                "'after hours'/'late night' -> [3, 5]; "
                "'closing'/'come-down' -> [2, 4]; "
                "'warm-up'/'chill' -> [1, 3]. "
                "Also call this for adjectives like 'banger', 'kicking', 'mellow', 'laid-back', 'deep'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {"type": "integer", "description": "Minimum energy 1-10"},
                    "max": {"type": "integer", "description": "Maximum energy 1-10"},
                },
                "required": ["min", "max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_bpm",
            "description": (
                "Call this ONLY when the user explicitly states a BPM or tempo number. "
                "e.g. '140 BPM', 'around 128', '130 to 135 bpm'. "
                "Do NOT infer BPM from genre alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {"type": "number", "description": "Minimum BPM"},
                    "max": {"type": "number", "description": "Maximum BPM"},
                },
                "required": ["min", "max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_track_count",
            "description": (
                "Call this ONLY when the user explicitly says how many tracks they want. "
                "e.g. 'give me 5 songs', 'I want 8 tracks', '3 tunes'. "
                "Do NOT call this if no count is mentioned — the default of 5 will be used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of tracks (1-20)"},
                },
                "required": ["count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_crate_direction",
            "description": (
                "Call this when the user specifies a direction relative to a previous crate or set. "
                "Examples: 'higher energy', 'different genre', 'bridge to techno', 'slow it down', "
                "'take it darker', 'maintain the vibe'. "
                "Use for branching crate generation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["energy_up", "energy_down", "genre_shift", "bridge", "maintain"],
                        "description": "The broad direction category",
                    },
                    "description": {
                        "type": "string",
                        "description": "Free-text description of the desired direction",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_track",
            "description": (
                "Call this when the user names a specific song or artist and wants similar tracks. "
                "Examples: 'songs like God\\'s Plan', 'find me tracks like Strobe', "
                "'more like Eric Prydz', 'something similar to that Bicep track'. "
                "Do NOT call this for genre/vibe descriptions like 'dark techno' or 'groovy house'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_name": {
                        "type": "string",
                        "description": "The track title, artist name, or combined reference the user mentioned",
                    },
                    "modifier": {
                        "type": "string",
                        "description": "Optional direction modifier like 'higher energy', 'darker', 'faster', or null if none",
                    },
                },
                "required": ["track_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Playlist tool definitions for LLM-driven playlist organisation.
# ---------------------------------------------------------------------------
_PLAYLIST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_playlist",
            "description": (
                "Create a new playlist and populate it with tracks matching the given criteria. "
                "Use this when the user asks to make a new playlist, folder, or crate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the new playlist",
                    },
                    "criteria": {
                        "type": "object",
                        "description": "Filter criteria for which tracks to include",
                        "properties": {
                            "genres": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Genre/style tags to match",
                            },
                            "vibes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Vibe/mood descriptors to match",
                            },
                            "energy_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[min, max] energy on 1-10 integer scale",
                            },
                            "bpm_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[min, max] BPM",
                            },
                            "max_tracks": {
                                "type": "integer",
                                "description": "Maximum tracks to add (default 40)",
                            },
                        },
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_playlist",
            "description": (
                "Add tracks matching criteria to an existing playlist. "
                "Use this when the user wants to add more tracks to a playlist that already exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_name": {
                        "type": "string",
                        "description": "Name of the existing playlist to add tracks to",
                    },
                    "criteria": {
                        "type": "object",
                        "description": "Filter criteria for which tracks to add",
                        "properties": {
                            "genres": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Genre/style tags to match",
                            },
                            "vibes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Vibe/mood descriptors to match",
                            },
                            "energy_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[min, max] energy on 1-10 integer scale",
                            },
                            "bpm_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[min, max] BPM",
                            },
                            "max_tracks": {
                                "type": "integer",
                                "description": "Maximum tracks to add (default 40)",
                            },
                        },
                    },
                },
                "required": ["playlist_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_all",
            "description": (
                "Organize the entire track library into multiple playlists at once. "
                "Use this when the user asks to sort, organize, or categorize all their tracks "
                "into folders/playlists/crates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playlists": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Playlist name",
                                },
                                "criteria": {
                                    "type": "object",
                                    "description": "Filter criteria for this playlist",
                                    "properties": {
                                        "genres": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "vibes": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "energy_range": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        "bpm_range": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        "max_tracks": {
                                            "type": "integer",
                                        },
                                    },
                                },
                            },
                            "required": ["name"],
                        },
                        "description": "List of playlists to create with their criteria",
                    },
                    "max_per_playlist": {
                        "type": "integer",
                        "description": "Default max tracks per playlist (default 40)",
                    },
                },
                "required": ["playlists"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_for_playlist",
            "description": (
                "Suggest tracks that would fit an existing playlist based on its current contents. "
                "Use this when the user asks for recommendations or suggestions for a playlist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_name": {
                        "type": "string",
                        "description": "Name of the playlist to suggest tracks for",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of suggestions (default 10)",
                    },
                },
                "required": ["playlist_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_tracks",
            "description": (
                "Move tracks matching criteria from one playlist to another. "
                "Removes from source and adds to target."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_playlist": {
                        "type": "string",
                        "description": "Name of the source playlist to move tracks FROM",
                    },
                    "target_playlist": {
                        "type": "string",
                        "description": "Name of the target playlist to move tracks TO (will be created if it doesn't exist)",
                    },
                    "criteria": {
                        "type": "object",
                        "description": "Optional criteria to filter which tracks to move. If omitted, moves ALL tracks.",
                        "properties": {
                            "genres": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Genre/style tags to match",
                            },
                            "vibes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Vibe/mood descriptors to match",
                            },
                            "energy_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[min, max] energy on 1-10 integer scale",
                            },
                            "bpm_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "[min_bpm, max_bpm]",
                            },
                        },
                    },
                },
                "required": ["source_playlist", "target_playlist"],
            },
        },
    },
]


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
        # Anthropic disabled — using Groq as primary provider
        self._anthropic_client = None
        self._anthropic_model = None

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
                "model": "gemini-2.5-flash"
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
                if tags_resp.data[:2]:
                    logger.info(f"DEBUG raw tag rows sample: {tags_resp.data[:2]}")
                    for r in tags_resp.data[:2]:
                        val = r.get("semantic_tags")
                        logger.info(f"DEBUG row semantic_tags type={type(val).__name__}, value={val!r}")
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
            logger.info(f"Tags: {sorted(tags.semantic_tags)}")
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

        Uses a single LLM tool-calling step that handles both intent detection
        and parameter extraction. The LLM chooses between find_similar_track
        (for "songs like X" queries) and the genre/vibe/energy tools (for
        descriptive queries). Transition intent is inferred from context.

        Returns a dict that always includes:
          - intent: str
          - track_count: int
          - confidence: float
          - reasoning: str
          For vibe_genre_search / transition_from_current:
            - tag_scores, vibe_scores, energy_range, bpm_range
          For find_similar_track:
            - track_name: str (the track the user asked about)
            - track_candidates: list of {trackid, title, artist, match_score}
        """
        if not self._tags_loaded:
            await self._load_available_tags()
        if not self.providers:
            return await self._fallback_interpretation(natural_query, context)

        # Determine if this is a transition request (current track context exists
        # and the query references it directionally).
        has_current_track = bool(context and context.current_track)
        intent = "vibe_genre_search"  # default; overridden below if tools say otherwise

        # ── Single LLM call: tool-calling with all tools (including find_similar_track) ──
        user_prompt = self._build_user_prompt(natural_query)
        try:
            system_prompt = self._build_system_prompt_tools(context, intent=intent)
            params = await self._generate_with_tools(system_prompt, user_prompt)

            # If the LLM called find_similar_track, route to similarity path
            if params.get("_find_similar_track"):
                fst = params["_find_similar_track"]
                intent_result = {
                    "track_name": fst.get("track_name", ""),
                    "modifier": fst.get("modifier"),
                    "model_used": params.get("model_used", "tool-calling"),
                }
                logger.info(f"🎯 Intent: find_similar_track — '{natural_query}'")
                try:
                    return await self._interpret_find_similar(
                        natural_query, intent_result, context,
                    )
                except Exception as e:
                    logger.error(f"find_similar_track failed: {e}")
                    result = await self._fallback_interpretation(natural_query, context)
                    result["intent"] = "find_similar_track"
                    return result

            # Determine intent from context: if current track is playing and
            # the LLM used crate_direction or the query is context-relative,
            # treat as transition.
            if has_current_track and params.get("crate_direction"):
                intent = "transition_from_current"

            params["intent"] = intent
            params.setdefault("track_count", 5)
            params.setdefault("confidence",  0.9)
            logger.info(f"🎯 Intent: {intent} — '{natural_query}'")
            # Widen BPM range if too narrow
            if params.get("bpm_range"):
                lo, hi = params["bpm_range"]
                if hi - lo < 6:
                    mid = (lo + hi) / 2
                    params["bpm_range"] = [max(1, mid - 4), mid + 4]
            return params
        except Exception as e:
            logger.warning(f"Tool-calling failed ({e}), falling back to JSON interpretation")

        # Fallback path — single JSON blob (old approach)
        try:
            system_prompt = self._build_system_prompt(context, intent=intent)
            parsed    = await self._generate_with_fallback(system_prompt, user_prompt)
            validated = await self._validate_and_enhance(parsed, context)
            validated["intent"] = intent
            return validated
        except Exception as e:
            logger.error(f"All LLM providers failed: {e}")
            result = await self._fallback_interpretation(natural_query, context)
            result["intent"] = intent
            return result

    # -------------------------------------------------------------------------
    # Public: interpret a playlist organisation command
    # -------------------------------------------------------------------------

    async def interpret_playlist_command(
        self,
        query: str,
        existing_playlists: list,
        track_summary: dict,
    ) -> list:
        """
        Convert a natural-language playlist organisation request into a list
        of structured actions using ``_PLAYLIST_TOOLS``.

        Returns a list of dicts, each like:
            {"tool": "create_playlist", "args": {"name": "...", "criteria": {...}}}
        """
        if not self._tags_loaded:
            await self._load_available_tags()

        semantic_tags_list = sorted(self.available_tags.semantic_tags)
        vibes_list = sorted(self.available_tags.vibes)

        # Build existing-playlists block
        playlist_lines = ""
        if existing_playlists:
            for p in existing_playlists:
                name = p.get("name", "Unnamed")
                count = p.get("track_count", 0)
                playlist_lines += f"- {name} ({count} tracks)\n"
        else:
            playlist_lines = "(none)\n"

        total_tracks = track_summary.get("total_tracks", 0)

        system_prompt = (
            "You are a DJ playlist organiser assistant. You help organize music "
            "tracks into playlists/folders.\n\n"
            f"Available genres/styles: {', '.join(semantic_tags_list)}\n\n"
            f"Available vibes: {', '.join(vibes_list)}\n\n"
            f"Track library: {total_tracks} tracks\n\n"
            "Existing playlists:\n"
            f"{playlist_lines}\n"
            "Use the provided tools to organize tracks. You can create playlists, "
            "add tracks to existing ones, organize all tracks, or suggest tracks "
            "for a playlist.\n\n"
            "RULES:\n"
            "- Genre/style names in criteria.genres MUST match entries from the "
            "available genres list exactly.\n"
            "- Vibe names in criteria.vibes MUST match entries from the available "
            "vibes list exactly.\n"
            "- Energy values are on a 0.0-1.0 scale.\n"
            "- Choose an appropriate number of tracks based on context. "
            "For a full playlist, use 12-25 tracks. For suggestions, use "
            "8-15 tracks. Minimum 5 tracks. Do not hardcode to exactly 5.\n"
            "- You may call multiple tools in one response."
        )

        user_prompt = f'Request: "{query}"'

        # Use the provider chain with _PLAYLIST_TOOLS
        actions = await self._generate_with_playlist_tools(system_prompt, user_prompt)
        return actions

    async def _generate_with_playlist_tools_claude(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, Any]]:
        """Use Claude for playlist tool calling (primary provider)."""
        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in _PLAYLIST_TOOLS:
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            })

        response = await self._anthropic_client.messages.create(
            model=self._anthropic_model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=anthropic_tools,
        )

        actions = []
        for block in response.content:
            if block.type == "tool_use":
                actions.append({"tool": block.name, "args": block.input})
        return actions

    async def _generate_with_playlist_tools(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Call the LLM with ``_PLAYLIST_TOOLS``. Returns a list of action dicts.
        Falls back provider-by-provider on failure.
        """
        # Try Claude first if available
        if self._anthropic_client:
            try:
                actions = await asyncio.wait_for(
                    self._generate_with_playlist_tools_claude(system_prompt, user_prompt),
                    timeout=30,
                )
                if actions:
                    logger.info(f"Claude returned {len(actions)} playlist actions")
                    return actions
            except Exception as e:
                logger.warning(f"Claude playlist tool calling failed: {e}, falling back to other providers")

        # Existing provider chain continues below...
        attempts = 0
        current_idx = self.active_provider_index

        while attempts < len(self.providers):
            provider = self.providers[current_idx]
            try:
                response = await provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools=_PLAYLIST_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    timeout=20.0,
                )
                msg = response.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    logger.warning(
                        f"{provider['name']} returned no playlist tool calls — skipping"
                    )
                    current_idx = (current_idx + 1) % len(self.providers)
                    attempts += 1
                    continue

                if current_idx != self.active_provider_index:
                    logger.info(f"Switched active provider to {provider['name']}")
                    self.active_provider_index = current_idx

                # Parse each tool call into an action dict
                actions = []
                for call in tool_calls:
                    try:
                        args = json.loads(call.function.arguments)
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.warning(f"Could not parse playlist tool args: {e}")
                        continue
                    actions.append({
                        "tool": call.function.name,
                        "args": args,
                    })

                logger.info(
                    f"Playlist tools called: "
                    f"{[a['tool'] for a in actions]} "
                    f"via {provider['name']}"
                )
                return actions

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                logger.warning(
                    f"{provider['name']} unavailable ({type(e).__name__}), switching"
                )
            except Exception as e:
                logger.warning(
                    f"{provider['name']} playlist tool-call error: {e}, switching"
                )

            current_idx = (current_idx + 1) % len(self.providers)
            attempts += 1

        raise RuntimeError("All providers failed playlist tool calling.")

    # -------------------------------------------------------------------------
    # Intent classification
    # -------------------------------------------------------------------------

    async def _classify_intent(
            self,
            query: str,
            context: Optional[InterpretationContext],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to classify the user's intent into one of three categories.
        Uses a small, focused prompt to keep latency low.
        """
        has_current_track = bool(context and context.current_track)
        current_track_hint = ""
        if has_current_track:
            t = context.current_track
            current_track_hint = (
                f"The DJ is currently playing: {t.get('title','?')} by {t.get('artist','?')}."
            )

        system_prompt = f"""You are a DJ assistant. Classify the user's request into exactly one intent.

{current_track_hint}

INTENTS:
1. "find_similar_track" — user names a specific song/artist and wants similar tracks.
   Examples: "songs like God's Plan", "find me tracks like Strobe", "more like Eric Prydz",
             "something like that last track", "give me stuff similar to Aphex Twin"

2. "vibe_genre_search" — user describes a sound, genre, vibe, or energy they want.
   Examples: "dark tech house", "upbeat minimal", "give me 6 ketty bangers",
             "something melodic and deep", "peak time euphoric techno"

3. "transition_from_current" — user references the current track and wants to move
   in a direction (higher/lower energy, different vibe, same key etc).
   Examples: "take it higher", "bring it down a bit", "keep this vibe but darker",
             "something that flows from this". Only valid if a current track is playing.

OUTPUT — valid JSON only:
{{
  "intent": "find_similar_track" | "vibe_genre_search" | "transition_from_current",
  "track_name": "<name of track/artist if intent is find_similar_track, else null>",
  "modifier": "<any energy/vibe direction modifier from the query, e.g. 'higher energy', 'darker', 'faster', 'more melodic', or null if none>",
  "reasoning": "<one sentence>"
}}"""

        user_prompt = f'Request: "{query}"'

        try:
            parsed = await self._generate_with_fallback(system_prompt, user_prompt)
            intent = parsed.get("intent", "vibe_genre_search")
            # If no current track is playing, transition_from_current is meaningless
            if intent == "transition_from_current" and not has_current_track:
                intent = "vibe_genre_search"
                parsed["reasoning"] = "No current track playing — treating as vibe/genre search."
            parsed["intent"] = intent
            return parsed
        except Exception as e:
            logger.warning(f"Intent classification failed: {e} — defaulting to vibe_genre_search")
            return {"intent": "vibe_genre_search", "track_name": None, "reasoning": "fallback"}

    # -------------------------------------------------------------------------
    # find_similar_track path: fuzzy name search → candidate list
    # -------------------------------------------------------------------------

    async def _interpret_find_similar(
            self,
            query: str,
            intent_result: Dict[str, Any],
            context: Optional[InterpretationContext],
    ) -> Dict[str, Any]:
        """
        Handle a find_similar_track intent.

        1. Extract the track/artist name from the intent classification.
        2. Fuzzy-search the DB for up to 5 candidate matches.
        3. Return a result with intent='find_similar_track' and track_candidates
           so the caller can pick the best match and run embedding similarity.
        """
        track_name = (intent_result.get("track_name") or "").strip()
        if not track_name:
            # Try to extract from raw query as a fallback
            track_name = query

        candidates = await self._fuzzy_track_search(track_name)

        import re
        m = re.search(r'\b(\d+)\s*(?:track|song|result|tune)s?\b', query.lower())
        track_count = max(1, int(m.group(1))) if m else 7

        modifier = (intent_result.get("modifier") or "").strip() or None

        return {
            "intent": "find_similar_track",
            "track_name": track_name,
            "track_candidates": candidates,
            "track_count": track_count,
            "modifier": modifier,
            "confidence": 0.95 if candidates else 0.3,
            "reasoning": (
                f"Looking for tracks similar to '{track_name}'"
                + (f" ({modifier})" if modifier else "")
                + f". Found {len(candidates)} candidate match(es) in library."
                if candidates
                else f"Could not find '{track_name}' in the library."
            ),
            "model_used": intent_result.get("model_used", "intent-classifier"),
        }

    async def _fuzzy_track_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the tracks table for title/artist matches using ilike (case-insensitive
        substring match). Returns a ranked list of candidates with a rough match_score.

        Postgres pg_trgm would be ideal but requires an extension; ilike on both
        fields covers the common case well enough without any schema changes.
        """
        query = query.strip()
        if not query:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}

        # Title search — exact substring match (highest confidence)
        try:
            resp = self.supabase.table("tracks") \
                .select("trackid, title, artist, bpm, key, album_art_url") \
                .ilike("title", f"%{query}%") \
                .limit(limit) \
                .execute()
            for row in (resp.data or []):
                tid = str(row["trackid"])
                # Score: 1.0 if query matches full title, else 0.85
                score = 1.0 if row.get("title", "").lower() == query.lower() else 0.85
                candidates[tid] = {**row, "trackid": tid, "match_score": score, "match_field": "title"}
        except Exception as e:
            logger.warning(f"Title search failed: {e}")

        # Artist search — slightly lower confidence
        if len(candidates) < limit:
            try:
                resp = self.supabase.table("tracks") \
                    .select("trackid, title, artist, bpm, key, album_art_url") \
                    .ilike("artist", f"%{query}%") \
                    .limit(limit) \
                    .execute()
                for row in (resp.data or []):
                    tid = str(row["trackid"])
                    if tid not in candidates:
                        candidates[tid] = {**row, "trackid": tid, "match_score": 0.75, "match_field": "artist"}
            except Exception as e:
                logger.warning(f"Artist search failed: {e}")

        # Try individual words if no results yet (handles "Gods Plan" → "God's Plan")
        if not candidates:
            words = [w for w in query.split() if len(w) > 3]
            for word in words[:3]:
                try:
                    resp = self.supabase.table("tracks") \
                        .select("trackid, title, artist, bpm, key, album_art_url") \
                        .ilike("title", f"%{word}%") \
                        .limit(limit) \
                        .execute()
                    for row in (resp.data or []):
                        tid = str(row["trackid"])
                        if tid not in candidates:
                            candidates[tid] = {**row, "trackid": tid, "match_score": 0.6, "match_field": "title_word"}
                except Exception as e:
                    logger.warning(f"Word search failed for '{word}': {e}")
                if candidates:
                    break

        result = sorted(candidates.values(), key=lambda x: x["match_score"], reverse=True)
        logger.info(f"🔍 Fuzzy search '{query}' → {len(result)} candidate(s)")
        return result[:limit]

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

            # Combine ALL tag + vibe conditions into a single .or_() call.
            # Two separate .or_() calls get ANDed by PostgREST, which would
            # require a track to match BOTH a genre tag AND a vibe — wrong.
            # One combined call means "match any tag OR any vibe".
            all_or_parts: list[str] = []
            if query_tags:
                all_or_parts += [
                    f'semantic_tags.cs.{json.dumps([t])}' for t in query_tags
                ]
            if query_vibes:
                all_or_parts += [
                    f'vibe.cs.{json.dumps([v])}' for v in query_vibes
                ]
            if all_or_parts:
                labels_q = labels_q.or_(",".join(all_or_parts))

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
                .select("trackid, title, artist, bpm, key, filepath, album_art_url, embedding")
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

        # Dynamic weights: if request is vibe-only, vibe should dominate.
        # If request is genre-only, tags should dominate.
        total_tag_confidence  = sum(tag_scores.values())  if has_tags  else 0.0
        total_vibe_confidence = sum(vibe_scores.values()) if has_vibes else 0.0
        if has_vibes and not has_tags:
            tag_weight  = 0.10
            vibe_weight = 0.65
        elif has_tags and not has_vibes:
            tag_weight  = 0.60
            vibe_weight = 0.10
        elif has_tags and has_vibes and total_vibe_confidence > total_tag_confidence:
            # Vibe-led mixed query
            tag_weight  = 0.30
            vibe_weight = 0.50
        else:
            # Genre-led or balanced mixed query
            tag_weight  = 0.45
            vibe_weight = 0.35

        scored = []
        for t in tracks:
            score = 0.0

            # ── Tag score ─────────────────────────────────────────────────────
            if has_tags:
                track_tags = [tg.lower() for tg in (t.get("semantic_tags") or [])]
                tag_hit    = sum(tag_scores_lower.get(tg, 0.0) for tg in track_tags)
                score     += (tag_hit / max_tag_score) * tag_weight

            # ── Vibe score ────────────────────────────────────────────────────
            if has_vibes:
                tv = t.get("vibe") or []
                track_vibes = [v.lower() for v in (tv if isinstance(tv, list) else [tv])]
                vibe_hit    = sum(vibe_scores_lower.get(v, 0.0) for v in track_vibes)
                score      += (vibe_hit / max_vibe_score) * vibe_weight

            # ── Energy proximity (0–0.12) ─────────────────────────────────────
            if has_energy and t.get("energy") is not None:
                lo, hi = energy_range
                mid    = (lo + hi) / 2
                span   = max(hi - lo, 1)
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
                .select("trackid, title, artist, bpm, key, filepath, album_art_url, embedding")
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

    async def _generate_with_tools(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Call the LLM with tool-calling enabled. The LLM decides which tools to
        invoke based on what the query actually needs — no required fields.
        Tries Anthropic (Claude) first, then falls back provider-by-provider.
        """
        # ── Try Claude first (best at tool-calling) ──
        if self._anthropic_client:
            try:
                anthropic_tools = []
                for tool in _SEARCH_TOOLS:
                    fn = tool["function"]
                    anthropic_tools.append({
                        "name": fn["name"],
                        "description": fn["description"],
                        "input_schema": fn["parameters"],
                    })
                response = await asyncio.wait_for(
                    self._anthropic_client.messages.create(
                        model=self._anthropic_model,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        tools=anthropic_tools,
                    ),
                    timeout=15,
                )
                # Convert Anthropic tool_use blocks to the same format as OpenAI
                tool_calls_raw = [b for b in response.content if b.type == "tool_use"]
                if tool_calls_raw:
                    params = self._parse_tool_calls_anthropic(tool_calls_raw)
                    params["model_used"] = f"Claude ({self._anthropic_model})"
                    params["confidence"] = 0.9
                    params["reasoning"] = (
                        f"Tools called: {', '.join(b.name for b in tool_calls_raw)}"
                    )
                    logger.info(f"Claude search tool-calling succeeded: {[b.name for b in tool_calls_raw]}")
                    return params
                else:
                    logger.warning("Claude returned no search tool calls, falling back")
            except Exception as e:
                logger.warning(f"Claude search tool-calling failed: {e}, falling back")

        # ── Fallback: OpenAI-compatible providers ──
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
                    tools=_SEARCH_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    timeout=15.0,
                )
                msg        = response.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    logger.warning(f"⚠️  {provider['name']} returned no tool calls — skipping")
                    current_idx = (current_idx + 1) % len(self.providers)
                    attempts += 1
                    continue

                if current_idx != self.active_provider_index:
                    logger.info(f"✅ Switched active provider to {provider['name']}")
                    self.active_provider_index = current_idx

                params = self._parse_tool_calls(tool_calls)
                params["model_used"] = f"{provider['name']} ({provider['model']})"
                params["confidence"] = 0.9
                params["reasoning"]  = (
                    f"Tools called: {', '.join(c.function.name for c in tool_calls)}"
                )
                return params

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                logger.warning(f"⚠️  {provider['name']} unavailable ({type(e).__name__}), switching…")
            except Exception as e:
                logger.warning(f"⚠️  {provider['name']} tool-call error: {e}, switching…")

            current_idx = (current_idx + 1) % len(self.providers)
            attempts   += 1

        raise RuntimeError("All providers failed tool calling.")

    def _parse_tool_calls(self, tool_calls) -> Dict[str, Any]:
        """
        Convert LLM tool calls into the search params dict consumed by search().
        Handles fuzzy rescue for names not in DB, and normalises energy to 0-1.
        """
        tag_lower  = {t.lower(): t for t in self.available_tags.semantic_tags}
        vibe_lower = {v.lower(): v for v in self.available_tags.vibes}

        logger.info(f"DEBUG tag_lower keys ({len(tag_lower)}): {sorted(tag_lower.keys())}")
        logger.info(f"DEBUG available_tags.semantic_tags ({len(self.available_tags.semantic_tags)}): {sorted(self.available_tags.semantic_tags)}")

        params: Dict[str, Any] = {
            "tag_scores":    {},
            "vibe_scores":   {},
            "energy_range":  None,
            "bpm_range":     None,
            "track_count":   5,
        }

        for call in tool_calls:
            fn   = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"  Could not parse args for {fn}: {e}")
                continue

            if fn == "set_genre":
                for g in args.get("genres", []):
                    name = g.get("name", "")
                    conf = float(g.get("confidence", 0.5))
                    name_lower = name.lower()
                    canonical = tag_lower.get(name_lower)
                    logger.info(f"DEBUG set_genre: LLM sent name={name!r}, name.lower()={name_lower!r}, lookup result={canonical!r}, in tag_lower={name_lower in tag_lower}")
                    if canonical:
                        params["tag_scores"][canonical] = round(conf, 3)
                    else:
                        rescued = self._fuzzy_rescue(name, tag_lower, conf)
                        if rescued:
                            canon, score = rescued
                            params["tag_scores"][canon] = max(
                                params["tag_scores"].get(canon, 0.0), score
                            )
                        else:
                            logger.warning(f"  set_genre: '{name}' not in DB, discarded")

            elif fn == "set_vibe":
                for v in args.get("vibes", []):
                    name = v.get("name", "")
                    conf = float(v.get("confidence", 0.5))
                    canonical = vibe_lower.get(name.lower())
                    if canonical:
                        params["vibe_scores"][canonical] = round(conf, 3)
                    else:
                        rescued = self._fuzzy_rescue(name, vibe_lower, conf)
                        if rescued:
                            canon, score = rescued
                            params["vibe_scores"][canon] = max(
                                params["vibe_scores"].get(canon, 0.0), score
                            )
                        else:
                            logger.warning(f"  set_vibe: '{name}' not in DB, discarded")

            elif fn == "set_energy":
                lo = int(round(float(args.get("min", 5))))
                hi = int(round(float(args.get("max", 7))))
                # Clamp to DB scale [1, 10]
                params["energy_range"] = [
                    max(1, min(10, lo)),
                    max(1, min(10, hi)),
                ]

            elif fn == "set_bpm":
                params["bpm_range"] = [
                    float(args.get("min", 120)),
                    float(args.get("max", 140)),
                ]

            elif fn == "set_track_count":
                params["track_count"] = max(1, min(20, int(args.get("count", 5))))

            elif fn == "set_crate_direction":
                params["crate_direction"] = args.get("direction")
                params["crate_direction_desc"] = args.get("description", "")

            elif fn == "find_similar_track":
                params["_find_similar_track"] = {
                    "track_name": args.get("track_name", ""),
                    "modifier": args.get("modifier"),
                }

        # Flat lists for backwards compat with existing search/scoring code
        params["semantic_tags"] = list(params["tag_scores"].keys())
        params["vibes"]         = list(params["vibe_scores"].keys())
        params["interpretation_method"] = "tool_calls"

        logger.info(
            f"🛠️  Tool params — tags: {params['tag_scores']}, "
            f"vibes: {params['vibe_scores']}, "
            f"energy: {params['energy_range']}, "
            f"bpm: {params['bpm_range']}, "
            f"count: {params['track_count']}"
        )
        return params

    def _parse_tool_calls_anthropic(self, tool_use_blocks) -> Dict[str, Any]:
        """Convert Anthropic tool_use blocks to the same params dict as _parse_tool_calls."""
        # Create thin wrappers that match the OpenAI tool_call interface
        class _FnShim:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class _CallShim:
            def __init__(self, block):
                self.function = _FnShim(block.name, json.dumps(block.input))

        shims = [_CallShim(b) for b in tool_use_blocks]
        return self._parse_tool_calls(shims)

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------

    def _build_system_prompt_tools(
            self,
            context: Optional[InterpretationContext],
            intent: str = "vibe_genre_search",
    ) -> str:
        """
        Concise system prompt for the tool-calling path.
        The tool schemas carry the structural rules, so this just explains
        the DJ context and which tool to use for what.
        """
        semantic_tags_list = sorted(self.available_tags.semantic_tags)
        vibes_list         = sorted(self.available_tags.vibes)

        context_info       = ""
        transition_note    = ""
        if context and context.current_track:
            t = context.current_track
            context_info = (
                f"\nCurrently playing: {t.get('title','?')} by {t.get('artist','?')}"
                f"\n  Genre tags: {t.get('semantic_tags') or []}"
                f"  Vibes: {t.get('vibe_descriptors') or []}"
                f"  Energy: {t.get('energy','?')}  BPM: {t.get('bpm','?')}\n"
            )
            if intent == "transition_from_current":
                transition_note = (
                    "\nTRANSITION REQUEST: use the current track as your baseline. "
                    "Shift genre/vibe/energy in the direction the user indicates. "
                    "Preserve tags/vibes that are compatible with the new direction.\n"
                )

        return f"""You are an expert DJ assistant. Use the provided tools to define what music to search for.
{context_info}{transition_note}
AVAILABLE GENRES (use these exact names in set_genre):
{json.dumps(semantic_tags_list)}

AVAILABLE VIBES (use these exact names in set_vibe):
{json.dumps(vibes_list)}

RULES:
- If the user names a specific song or artist and wants similar tracks → call find_similar_track.
  Do NOT also call set_genre/set_vibe in that case.
- Genre words (house, techno, drum and bass, ambient…) → set_genre
- Adjective/mood/atmosphere words (dark, warm, bouncy, driving…) → set_vibe
- Genre and vibe are INDEPENDENT axes. Never put a vibe in set_genre or vice versa.
- Always include related genres/vibes at lower confidence for natural search widening.
- Energy comes from context words: "banger", "kicking", "peak time" → set_energy high;
  "chill", "warm-up", "after hours" → set_energy lower.
- Only call set_bpm when user explicitly states a BPM number.
- Only call set_track_count when user explicitly states how many tracks they want.
- If an aspect is not mentioned, do not call that tool."""

    def _build_system_prompt(
            self,
            context: Optional[InterpretationContext],
            intent: str = "vibe_genre_search",
    ) -> str:
        semantic_tags_list = sorted(self.available_tags.semantic_tags)
        vibes_list         = sorted(self.available_tags.vibes)

        context_info = ""
        transition_instruction = ""
        if context and context.current_track:
            t = context.current_track
            ctx_tags  = t.get("semantic_tags") or []
            ctx_vibes = t.get("vibe_descriptors") or []
            context_info = (
                f"\nCurrent Track: {t.get('title','Unknown')} by {t.get('artist','Unknown')}"
                f"\n  Key: {t.get('key','?')}  BPM: {t.get('bpm','?')}  Energy: {t.get('energy','?')}"
                f"\n  Tags: {ctx_tags}  Vibes: {ctx_vibes}\n"
            )
            if intent == "transition_from_current":
                transition_instruction = (
                    "\nThis is a TRANSITION request from the current track above. "
                    "Use the current track's tags/vibes/energy as the baseline and "
                    "shift them according to what the user asks (higher energy, darker, etc). "
                    "Preserve compatible tags unless the user explicitly wants to change them.\n"
                )

        return f"""You are an expert DJ assistant interpreting messy natural language requests.
{context_info}{transition_instruction}
AVAILABLE DATABASE TAGS — you MUST only use tags from these EXACT lists. No exceptions.
Genres/Styles : {json.dumps(semantic_tags_list)}
Vibes         : {json.dumps(vibes_list)}

TASK:
The user's request has TWO independent axes — GENRE and VIBE. Extract both separately.

GENRE AXIS → tag_scores:
  Every word that names a musical genre or style belongs here.
  Genres are things like: house, techno, drum and bass, ambient, acid, trance, minimal, etc.
  - The named genre gets a high score (0.85–1.0).
  - Also add closely related/neighbouring genres at lower scores (0.3–0.65) so the search
    can widen naturally when not enough exact matches are found. E.g. "house" → also include
    "tech house", "deep house", "garage" at lower scores.
  - NEVER put vibe/mood words in tag_scores.

VIBE AXIS → vibe_scores:
  Every adjective, mood, atmosphere, or feeling word belongs here.
  Vibes are things like: dark, bouncy, warm, hypnotic, groovy, driving, euphoric, etc.
  - The named vibe gets a high score (0.85–1.0).
  - Also add closely related vibes at lower scores (0.3–0.65) for widening. E.g. "dark" →
    also include "mysterious", "hypnotic", "moody" at lower scores.
  - NEVER put genre words in vibe_scores.
  - NEVER convert a vibe into a genre. "Warm" is NOT "house". "Dark" is NOT "techno".

THE TWO AXES ARE COMPLETELY INDEPENDENT. Do not let genre influence vibe_scores or vice versa.

CRITICAL RULES:
1. ONLY output names that appear VERBATIM in the Genres/Styles or Vibes lists above.
   Any name not in the list is silently discarded and hurts the search.
2. If a vibe word the user uses is not in the Vibes list, map it to the closest available vibe.
   Never leave vibe_scores empty when the user describes a mood — always approximate.
   e.g. "bittersweet" → "melancholic", "emotional"; "intense" → "driving", "aggressive"
3. If no genre is mentioned, leave tag_scores empty {{}}.
   If no vibe is mentioned, leave vibe_scores empty {{}}.
4. Score reflects how close the match is: exact match → 0.9+, neighbour → 0.4–0.65, rough → 0.2–0.4.
5. Energy range is on a 1–10 scale (1=very chill, 10=absolute peak time).
   Infer from context: "kicking"→[7,9], "banging"→[8,10], "peak time"→[8,10],
   "dark moody"→[4,7], "warm-up"→[2,5], "after hours"→[5,8], "bittersweet"→[4,6]
6. Examples of correct output:
   "kicking techno" →
     tag_scores:  {{"techno": 0.9, "minimal techno": 0.5, "industrial techno": 0.4}}
     vibe_scores: {{"driving": 0.85, "aggressive": 0.6, "energetic": 0.5}}
   "dark but bouncy house" →
     tag_scores:  {{"house": 0.9, "tech house": 0.6, "deep house": 0.4}}
     vibe_scores: {{"dark": 0.9, "bouncy": 0.85, "groovy": 0.5, "mysterious": 0.4}}
   "something warm and late night" (no genre) →
     tag_scores:  {{}}
     vibe_scores: {{"warm": 0.9, "late night": 0.85, "soulful": 0.5, "groovy": 0.4}}
7. "give me 6 tracks" → track_count: 6  |  no count mentioned → track_count: 5

OUTPUT — valid JSON only, no markdown:
{{
    "tag_scores"       : {{"tag_name": 0.0-1.0}},
    "vibe_scores"      : {{"vibe_name": 0.0-1.0}},
    "energy_range"     : [min, max] on 1-10 scale, or null,
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

    def _fuzzy_rescue(self, name: str, lower_map: Dict[str, str], score: float) -> Optional[Tuple[str, float]]:
        """
        When the LLM outputs a tag/vibe not in the DB, try to rescue it by finding
        the closest DB entry using substring and character-level similarity.
        Returns (canonical_name, adjusted_score) or None if no good match found.
        """
        from difflib import SequenceMatcher
        name_lower = name.lower()
        name_words = set(name_lower.split())

        best_match: Optional[str] = None
        best_sim = 0.0

        for db_lower, db_canonical in lower_map.items():
            # Substring containment (e.g. "emotional" ↔ "emotionally driven")
            if name_lower in db_lower or db_lower in name_lower:
                sim = 0.8
            else:
                # Jaccard on word tokens
                db_words = set(db_lower.split())
                union = name_words | db_words
                sim = len(name_words & db_words) / len(union) if union else 0.0
                # Character-level ratio as tiebreaker
                char_sim = SequenceMatcher(None, name_lower, db_lower).ratio()
                sim = max(sim, char_sim)

            if sim > best_sim:
                best_sim = sim
                best_match = db_canonical

        if best_match and best_sim >= 0.35:
            rescued_score = round(float(score) * best_sim, 3)
            logger.info(f"  ↩ Fuzzy rescued '{name}' → '{best_match}' (sim={best_sim:.2f}, score={rescued_score})")
            return best_match, rescued_score
        return None

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
                # Safety net: fuzzy match to closest DB tag
                rescued = self._fuzzy_rescue(tag, tag_lower_map, score)
                if rescued:
                    canon, adj_score = rescued
                    tag_scores[canon] = max(tag_scores.get(canon, 0.0), adj_score)
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
                # Safety net: fuzzy match to closest DB vibe
                rescued = self._fuzzy_rescue(vibe, vibe_lower_map, score)
                if rescued:
                    canon, adj_score = rescued
                    vibe_scores[canon] = max(vibe_scores.get(canon, 0.0), adj_score)
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