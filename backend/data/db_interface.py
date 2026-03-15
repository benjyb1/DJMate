"""
Enhanced Database interface for DJMate with ML-optimized operations.
Implements async patterns, connection pooling, caching, and vector operations.
"""

import os
import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from supabase import create_client, Client
import asyncpg
from redis import Redis
import hashlib

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrackMetadata:
    """Structured track metadata for type safety and validation."""
    trackid: str
    filepath: str
    title: str
    artist: str
    album: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    duration: Optional[float] = None
    embedding: Optional[List[float]] = None
    semantic_tags: Optional[List[str]] = None
    energy: Optional[float] = None
    vibe_descriptors: Optional[List[str]] = None
    confidence_score: Optional[float] = None

@dataclass
class SimilarityResult:
    """Structured similarity search result."""
    track: TrackMetadata
    similarity_score: float
    match_type: str  # 'embedding', 'semantic', 'hybrid'

class DatabaseManager:
    """
    Enhanced database manager with ML-optimized operations.
    Implements connection pooling, caching, and vector search optimization.
    """

    def __init__(self, enable_caching: bool = True, pool_size: int = 10):
        """Initialize with enhanced configuration."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.postgres_url = os.getenv("DATABASE_URL")  # Direct PostgreSQL connection

        # Initialize clients
        self.client: Optional[Client] = None
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[Redis] = None

        # Configuration
        self.enable_caching = enable_caching
        self.pool_size = pool_size
        self.cache_ttl = 3600  # 1 hour default TTL

        # Performance metrics
        self.query_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0
        }

        self._initialize_clients()

    def _initialize_clients(self):
        # Always define client first
        self.client = None

        if not self.supabase_url or not self.supabase_key:
            logger.warning("SUPABASE credentials not found.")
            return

        try:
            self.client = create_client(self.supabase_url, self.supabase_key)

            # Redis for caching (optional)
            if self.enable_caching:
                try:
                    self.redis_client = Redis(
                        host=os.getenv('REDIS_HOST', 'localhost'),
                        port=int(os.getenv('REDIS_PORT', 6379)),
                        decode_responses=True
                    )
                    self.redis_client.ping()  # Test connection
                    logger.info("✅ Redis cache connected")
                except Exception as e:
                    logger.warning(f"Redis cache unavailable: {e}")
                    self.redis_client = None

            logger.info("✅ Database clients initialized")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    async def initialize_async_pool(self):
        """Initialize async PostgreSQL connection pool for vector operations."""
        if not self.postgres_url:
            logger.warning("PostgreSQL URL not provided. Vector operations limited.")
            return

        try:
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=30
            )
            logger.info(f"✅ PostgreSQL pool initialized ({self.pool_size} connections)")
        except Exception as e:
            logger.error(f"❌ PostgreSQL pool initialization failed: {e}")

    @asynccontextmanager
    async def get_pg_connection(self):
        """Async context manager for PostgreSQL connections."""
        if not self.pg_pool:
            await self.initialize_async_pool()

        async with self.pg_pool.acquire() as connection:
            yield connection

    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate consistent cache keys."""
        key_data = json.dumps(kwargs, sort_keys=True)
        hash_suffix = hashlib.md5(key_data.encode()).hexdigest()[:8]
        return f"djmate:{prefix}:{hash_suffix}"

    async def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache."""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                self.query_metrics['cache_hits'] += 1
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None

    async def _set_cache(self, cache_key: str, data: Dict[str, Any], ttl: int = None):
        """Store data in cache."""
        if not self.redis_client:
            return

        try:
            self.redis_client.setex(
                cache_key,
                ttl or self.cache_ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")

    async def get_track_by_id(self, track_id: str) -> Optional[TrackMetadata]:
        """Retrieve a single track with enhanced caching and type safety."""
        cache_key = self._cache_key("track", track_id=track_id)

        # Check cache first
        cached_data = await self._get_cached(cache_key)
        if cached_data:
            return TrackMetadata(**cached_data)

        if not self.client:
            return None

        try:
            start_time = asyncio.get_event_loop().time()

            response = self.client.table("tracks") \
                .select("*, track_labels(semantic_tags, energy, vibe)") \
                .eq("trackid", track_id) \
                .single() \
                .execute()

            # Update metrics
            self.query_metrics['total_queries'] += 1
            response_time = asyncio.get_event_loop().time() - start_time
            self._update_avg_response_time(response_time)

            if not response.data:
                return None

            # Structure the data
            track_data = self._structure_track_data(response.data)
            track_metadata = TrackMetadata(**track_data)

            # Cache the result
            await self._set_cache(cache_key, asdict(track_metadata))

            return track_metadata

        except Exception as e:
            logger.error(f"Error fetching track {track_id}: {e}")
            return None

    async def get_tracks_by_ids(self, track_ids: List[str]) -> List[TrackMetadata]:
        """Batch retrieve tracks with optimized caching."""
        if not track_ids:
            return []

        # Check cache for each track
        cached_tracks = {}
        uncached_ids = []

        for track_id in track_ids:
            cache_key = self._cache_key("track", track_id=track_id)
            cached_data = await self._get_cached(cache_key)
            if cached_data:
                cached_tracks[track_id] = TrackMetadata(**cached_data)
            else:
                uncached_ids.append(track_id)

        # Fetch uncached tracks
        uncached_tracks = {}
        if uncached_ids and self.client:
            try:
                response = self.client.table("tracks") \
                    .select("*, track_labels(semantic_tags, energy, vibe)") \
                    .in_("trackid", uncached_ids) \
                    .execute()

                for track_data in response.data or []:
                    structured_data = self._structure_track_data(track_data)
                    track_metadata = TrackMetadata(**structured_data)
                    track_id = track_metadata.trackid

                    uncached_tracks[track_id] = track_metadata

                    # Cache individual tracks
                    cache_key = self._cache_key("track", track_id=track_id)
                    await self._set_cache(cache_key, asdict(track_metadata))

            except Exception as e:
                logger.error(f"Error batch fetching tracks: {e}")

        # Combine cached and uncached results in original order
        result = []
        for track_id in track_ids:
            if track_id in cached_tracks:
                result.append(cached_tracks[track_id])
            elif track_id in uncached_tracks:
                result.append(uncached_tracks[track_id])

        return result

    async def vector_similarity_search(
            self,
            query_embedding: List[float],
            limit: int = 20,
            threshold: float = 0.7,
            include_metadata: bool = True
    ) -> List[SimilarityResult]:
        """
        Optimized vector similarity search using direct PostgreSQL connection.
        """
        if not self.pg_pool:
            await self.initialize_async_pool()

        if not self.pg_pool:
            # Fallback to Supabase RPC
            return await self._fallback_similarity_search(query_embedding, limit, threshold)

        cache_key = self._cache_key(
            "similarity",
            embedding_hash=hashlib.md5(str(query_embedding).encode()).hexdigest()[:16],
            limit=limit,
            threshold=threshold
        )

        # Check cache
        cached_results = await self._get_cached(cache_key)
        if cached_results:
            return [SimilarityResult(**result) for result in cached_results]

        try:
            async with self.get_pg_connection() as conn:
                # Optimized vector similarity query
                query = """
                SELECT 
                    t.trackid,
                    t.title,
                    t.artist,
                    t.album,
                    t.bpm,
                    t.key,
                    t.filepath,
                    tl.semantic_tags,
                    tl.energy,
                    tl.vibe,
                    1 - (t.embedding <=> $1::vector) as similarity_score
                FROM tracks t
                LEFT JOIN track_labels tl ON t.trackid = tl.trackid
                WHERE t.embedding IS NOT NULL 
                AND 1 - (t.embedding <=> $1::vector) > $2
                ORDER BY t.embedding <=> $1::vector
                LIMIT $3;
                """

                rows = await conn.fetch(query, query_embedding, threshold, limit)

                results = []
                for row in rows:
                    track_data = {
                        'trackid': str(row['trackid']),
                        'title': row['title'],
                        'artist': row['artist'],
                        'album': row['album'],
                        'bpm': row['bpm'],
                        'key': row['key'],
                        'filepath': row['filepath'],
                        'semantic_tags': row['semantic_tags'] or [],
                        'energy': row['energy'],
                        'vibe_descriptors': row['vibe'] or []
                    }

                    track_metadata = TrackMetadata(**track_data)
                    similarity_result = SimilarityResult(
                        track=track_metadata,
                        similarity_score=float(row['similarity_score']),
                        match_type='embedding'
                    )
                    results.append(similarity_result)

                # Cache results
                cache_data = [asdict(result) for result in results]
                await self._set_cache(cache_key, cache_data, ttl=1800)  # 30 min TTL

                return results

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return await self._fallback_similarity_search(query_embedding, limit, threshold)

    async def get_tracks_by_semantic_filter(
            self,
            structured_query: Dict[str, Any],
            limit: int = 100
    ) -> List[TrackMetadata]:
        """Enhanced semantic filtering with caching and optimization."""

        cache_key = self._cache_key("semantic_filter", **structured_query, limit=limit)
        cached_results = await self._get_cached(cache_key)
        if cached_results:
            return [TrackMetadata(**track) for track in cached_results]

        if not self.client:
            return []

        try:
            # Build optimized query
            query = self.client.table("tracks").select(
                "trackid, filepath, title, artist, album, bpm, key, "
                "track_labels!inner(semantic_tags, energy, vibe)"
            )

            # Apply BPM filter at database level
            bpm_range = structured_query.get("bpm_range")
            if bpm_range and len(bpm_range) == 2:
                query = query.gte("bpm", bpm_range[0]).lte("bpm", bpm_range[1])

            # Apply energy filter at database level if possible
            energy_range = structured_query.get("energy_range")
            if energy_range and len(energy_range) == 2:
                query = query.gte("track_labels.energy", energy_range[0]) \
                    .lte("track_labels.energy", energy_range[1])

            response = query.limit(limit * 2).execute()  # Over-fetch for filtering

            if not response.data:
                return []

            # Post-process filtering
            results = []
            tags = set(structured_query.get("semantic_tags", []) or structured_query.get("tags", []))
            vibe_descriptors = set(structured_query.get("vibes", []) or structured_query.get("vibe_descriptors", []))

            for track in response.data:
                labels = track.get("track_labels", {})
                if not labels:
                    continue
                # Supabase inner joins return labels as a list — unwrap it
                if isinstance(labels, list):
                    labels = labels[0] if labels else {}

                # Tag filtering (semantic_tags may be a JSON string, not a list)
                if tags:
                    track_tags = set(self._parse_json_field(labels.get("semantic_tags")))
                    if not tags.intersection(track_tags):
                        continue

                # Vibe filtering (vibe may be a JSON string, not a list)
                if vibe_descriptors:
                    track_vibes = set(self._parse_json_field(labels.get("vibe")))
                    if not vibe_descriptors.intersection(track_vibes):
                        continue

                # Structure track data
                track_data = self._structure_track_data(track)
                track_metadata = TrackMetadata(**track_data)
                results.append(track_metadata)

                if len(results) >= limit:
                    break

            # Cache results
            cache_data = [asdict(track) for track in results]
            await self._set_cache(cache_key, cache_data, ttl=1800)

            return results

        except Exception as e:
            logger.error(f"Semantic filtering failed: {e}")
            return []

    async def hybrid_search(
            self,
            query_embedding: Optional[List[float]] = None,
            structured_query: Optional[Dict[str, Any]] = None,
            weights: Dict[str, float] = None,
            limit: int = 50
    ) -> List[SimilarityResult]:
        """
        Advanced hybrid search combining vector similarity and semantic filtering.
        """
        if weights is None:
            weights = {"embedding": 0.6, "semantic": 0.4}

        results = []

        # Vector similarity search
        if query_embedding and weights.get("embedding", 0) > 0:
            vector_results = await self.vector_similarity_search(
                query_embedding,
                limit=limit * 2,
                threshold=0.5
            )

            for result in vector_results:
                result.similarity_score *= weights["embedding"]
                result.match_type = "embedding"
                results.append(result)

        # Semantic search
        if structured_query and weights.get("semantic", 0) > 0:
            semantic_tracks = await self.get_tracks_by_semantic_filter(
                structured_query,
                limit=limit * 2
            )

            for track in semantic_tracks:
                # Calculate semantic similarity score
                semantic_score = self._calculate_semantic_score(track, structured_query)
                weighted_score = semantic_score * weights["semantic"]

                # Check if track already in results (from vector search)
                existing_result = next(
                    (r for r in results if r.track.trackid == track.trackid),
                    None
                )

                if existing_result:
                    # Combine scores
                    existing_result.similarity_score += weighted_score
                    existing_result.match_type = "hybrid"
                else:
                    # Add new result
                    similarity_result = SimilarityResult(
                        track=track,
                        similarity_score=weighted_score,
                        match_type="semantic"
                    )
                    results.append(similarity_result)

        # Sort by combined score and return top results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]

    def _calculate_semantic_score(
            self,
            track: TrackMetadata,
            structured_query: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity score based on tag overlap and other factors."""
        score = 0.0

        # Tag overlap score
        query_tags = set(structured_query.get("tags", []))
        track_tags = set(track.semantic_tags or [])
        if query_tags:
            tag_overlap = len(query_tags.intersection(track_tags)) / len(query_tags)
            score += tag_overlap * 0.4

        # Vibe overlap score
        query_vibes = set(structured_query.get("vibe_descriptors", []))
        track_vibes = set(track.vibe_descriptors or [])
        if query_vibes:
            vibe_overlap = len(query_vibes.intersection(track_vibes)) / len(query_vibes)
            score += vibe_overlap * 0.3

        # Energy proximity score
        energy_range = structured_query.get("energy_range")
        if energy_range and track.energy is not None:
            target_energy = sum(energy_range) / 2
            energy_diff = abs(track.energy - target_energy)
            energy_score = max(0, 1 - energy_diff * 2)  # Normalize to 0-1
            score += energy_score * 0.3

        return min(score, 1.0)  # Cap at 1.0

    @staticmethod
    def _parse_json_field(value) -> list:
        """Parse a field that may be a JSON-encoded string or already a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            if value.strip():
                return [value.strip()]
        return []

    def _structure_track_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure raw database response into TrackMetadata format."""
        labels = raw_data.get("track_labels", {})
        if isinstance(labels, list) and labels:
            labels = labels[0]  # Take first label if multiple
        if not isinstance(labels, dict):
            labels = {}

        return {
            "trackid": raw_data["trackid"],
            "filepath": raw_data.get("filepath"),
            "title": raw_data["title"],
            "artist": raw_data["artist"],
            "album": raw_data.get("album"),
            "bpm": raw_data.get("bpm"),
            "key": raw_data.get("key"),
            "duration": raw_data.get("duration"),
            "embedding": raw_data.get("embedding"),
            "semantic_tags": self._parse_json_field(labels.get("semantic_tags")),
            "energy": labels.get("energy"),
            "vibe_descriptors": self._parse_json_field(labels.get("vibe")),
        }

    async def _fallback_similarity_search(
            self,
            query_embedding: List[float],
            limit: int,
            threshold: float
    ) -> List[SimilarityResult]:
        """Fallback similarity search using Supabase RPC."""
        if not self.client:
            return []

        try:
            response = self.client.rpc('match_tracks', {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }).execute()

            results = []
            for row in response.data or []:
                track_data = self._structure_track_data(row)
                track_metadata = TrackMetadata(**track_data)

                similarity_result = SimilarityResult(
                    track=track_metadata,
                    similarity_score=row.get('similarity', 0.0),
                    match_type='embedding'
                )
                results.append(similarity_result)

            return results

        except Exception as e:
            logger.error(f"Fallback similarity search failed: {e}")
            return []

    def _update_avg_response_time(self, response_time: float):
        """Update average response time metric."""
        total_queries = self.query_metrics['total_queries']
        current_avg = self.query_metrics['avg_response_time']

        self.query_metrics['avg_response_time'] = (
                (current_avg * (total_queries - 1) + response_time) / total_queries
        )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        cache_hit_rate = 0.0
        if self.query_metrics['total_queries'] > 0:
            cache_hit_rate = self.query_metrics['cache_hits'] / self.query_metrics['total_queries']

        return {
            **self.query_metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_enabled': self.redis_client is not None,
            'pg_pool_active': self.pg_pool is not None
        }

    async def find_similar_tracks(
            self,
            query_embedding: List[float],
            limit: int = 20,
            threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find tracks with similar embeddings. Returns plain dicts for easy
        serialisation — used by the /tracks/{id}/neighbors endpoint.
        """
        if not self.client:
            return []

        try:
            response = self.client.rpc('match_tracks', {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }).execute()

            results = []
            for row in response.data or []:
                results.append({
                    "id":         row.get("trackid") or row.get("id"),
                    "trackid":    row.get("trackid") or row.get("id"),
                    "title":      row.get("title", "Unknown"),
                    "artist":     row.get("artist", "Unknown"),
                    "bpm":        row.get("bpm"),
                    "key":        row.get("key"),
                    "filepath":   row.get("filepath"),  # Added for audio playback
                    "similarity": row.get("similarity", 0.0),
                })
            return results

        except Exception as e:
            logger.error(f"find_similar_tracks failed: {e}")
            return []

    async def get_session_context(
            self,
            session_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve session context. TODO: implement sessions table in Supabase."""
        if not session_id:
            return None
        return {
            "session_id":     session_id,
            "previous_tracks": [],
            "preferences":    {}
        }

    async def update_crate(
            self,
            session_id: str,
            tracks: List[str],
            sequence: List[int],
            validation_results: Dict[str, Any]
    ) -> bool:
        """Persist crate. TODO: implement crates table in Supabase."""
        logger.info(f"Crate updated for session {session_id} — {len(tracks)} tracks, "
                    f"score={validation_results.get('overall_score', 0.0):.2f}")
        return True

    async def get_track_labels(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get labels for a single track from track_labels table."""
        if not self.client:
            return None

        try:
            response = self.client.table("track_labels") \
                .select("*") \
                .eq("trackid", track_id) \
                .single() \
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching track labels for {track_id}: {e}")
            return None

    async def update_track_labels(
        self,
        track_id: str,
        updates: Dict[str, Any],
        log_corrections: bool = True,
    ) -> Dict[str, Any]:
        """
        Update track labels and optionally log corrections.

        1. Fetch current labels
        2. If log_corrections, insert correction rows for changed fields
        3. Upsert track_labels with tag_source='manual'
        4. Invalidate cache for this track
        """
        if not self.client:
            raise RuntimeError("Supabase client not initialised")

        # 1. Fetch current labels (may be None if row doesn't exist yet)
        current = await self.get_track_labels(track_id)

        # 2. Log corrections for each changed field
        if log_corrections and current:
            for field in ("semantic_tags", "vibe", "energy"):
                if field not in updates:
                    continue
                old_value = current.get(field)
                new_value = updates[field]
                # Parse old value so comparison is apples-to-apples
                if field in ("semantic_tags", "vibe"):
                    old_value = self._parse_json_field(old_value)
                if old_value == new_value:
                    continue
                correction = {
                    "trackid": track_id,
                    "field": field,
                    "old_value": json.dumps(old_value),
                    "new_value": json.dumps(new_value),
                }
                try:
                    self.client.table("tag_corrections").insert(correction).execute()
                except Exception as e:
                    # Table may not exist yet — warn but don't block the update
                    logger.warning(f"Failed to log tag correction: {e}")

        # 3. Upsert track_labels
        payload = {"trackid": track_id, "tag_source": updates.get("tag_source", "manual")}
        for field in ("semantic_tags", "vibe", "energy"):
            if field in updates:
                payload[field] = updates[field]

        self.client.table("track_labels").upsert(payload).execute()

        # 4. Invalidate cache for this track
        cache_key = self._cache_key("track", track_id=track_id)
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception:
                pass

        # Return the updated row
        updated = await self.get_track_labels(track_id)
        return updated or payload

    async def get_available_tags(self) -> Dict[str, List[str]]:
        """Get all unique semantic_tags and vibes from the database."""
        if not self.client:
            return {"semantic_tags": [], "vibes": []}

        try:
            response = self.client.table("track_labels") \
                .select("semantic_tags, vibe") \
                .execute()

            all_tags: set = set()
            all_vibes: set = set()
            for row in response.data or []:
                for tag in self._parse_json_field(row.get("semantic_tags")):
                    all_tags.add(tag)
                for vibe in self._parse_json_field(row.get("vibe")):
                    all_vibes.add(vibe)

            return {
                "semantic_tags": sorted(all_tags),
                "vibes": sorted(all_vibes),
            }
        except Exception as e:
            logger.error(f"get_available_tags failed: {e}")
            return {"semantic_tags": [], "vibes": []}

    # ── Crate operations ─────────────────────────────────────────────────────

    async def create_crate_session(self, session_id: str, name: str) -> Dict[str, Any]:
        """Create a new crate session."""
        if not self.client:
            return {"id": session_id, "name": name, "crates": []}
        try:
            payload = {"id": session_id, "name": name}
            self.client.table("crate_sessions").insert(payload).execute()
            return {"id": session_id, "name": name, "crates": []}
        except Exception as e:
            logger.warning(f"create_crate_session DB insert failed (table may not exist): {e}")
            return {"id": session_id, "name": name, "crates": []}

    async def get_crate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a crate session with all its crates."""
        if not self.client:
            return None
        try:
            response = self.client.table("crate_sessions") \
                .select("*").eq("id", session_id).single().execute()
            if not response.data:
                return None
            # Fetch all crates for this session
            crates_resp = self.client.table("crates") \
                .select("*").eq("session_id", session_id) \
                .order("position").execute()
            crates = []
            for c in (crates_resp.data or []):
                crate = await self.get_crate(c["id"])
                if crate:
                    crates.append(crate)
            return {
                "id": session_id,
                "name": response.data.get("name", ""),
                "crates": crates,
            }
        except Exception as e:
            logger.warning(f"get_crate_session failed: {e}")
            return None

    async def save_crate(self, crate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a generated crate to the database."""
        if not self.client:
            return crate_data
        try:
            # Save crate metadata
            crate_row = {
                "id": crate_data["id"],
                "session_id": crate_data["session_id"],
                "parent_crate_id": crate_data.get("parent_crate_id"),
                "label": crate_data.get("label", ""),
                "avg_bpm": crate_data.get("metadata", {}).get("avg_bpm"),
                "avg_energy": crate_data.get("metadata", {}).get("avg_energy"),
                "dominant_key": crate_data.get("metadata", {}).get("dominant_key"),
                "dominant_tags": crate_data.get("metadata", {}).get("dominant_tags", []),
            }
            self.client.table("crates").upsert(crate_row).execute()

            # Save crate tracks
            tracks = crate_data.get("tracks", [])
            for i, t in enumerate(tracks):
                track_row = {
                    "crate_id": crate_data["id"],
                    "trackid": t.get("trackid") or t.get("id"),
                    "position": i,
                }
                self.client.table("crate_tracks").upsert(track_row).execute()

            return crate_data
        except Exception as e:
            logger.warning(f"save_crate failed (tables may not exist): {e}")
            return crate_data

    async def get_crate(self, crate_id: str) -> Optional[Dict[str, Any]]:
        """Get a crate with its tracks."""
        if not self.client:
            return None
        try:
            resp = self.client.table("crates") \
                .select("*").eq("id", crate_id).single().execute()
            if not resp.data:
                return None

            # Get tracks
            tracks_resp = self.client.table("crate_tracks") \
                .select("trackid, position") \
                .eq("crate_id", crate_id) \
                .order("position").execute()

            track_ids = [r["trackid"] for r in (tracks_resp.data or [])]
            tracks = await self.get_tracks_by_ids(track_ids)
            track_dicts = []
            for t in tracks:
                d = {
                    "trackid": t.trackid, "title": t.title, "artist": t.artist,
                    "bpm": t.bpm, "key": t.key, "energy": t.energy,
                    "semantic_tags": t.semantic_tags or [],
                    "vibe_descriptors": t.vibe_descriptors or [],
                }
                track_dicts.append(d)

            return {
                "id": crate_id,
                "session_id": resp.data.get("session_id"),
                "parent_crate_id": resp.data.get("parent_crate_id"),
                "label": resp.data.get("label", ""),
                "tracks": track_dicts,
                "metadata": {
                    "avg_bpm": resp.data.get("avg_bpm"),
                    "avg_energy": resp.data.get("avg_energy"),
                    "dominant_key": resp.data.get("dominant_key"),
                    "dominant_tags": resp.data.get("dominant_tags", []),
                    "track_count": len(track_dicts),
                },
            }
        except Exception as e:
            logger.warning(f"get_crate failed: {e}")
            return None

    async def update_crate_tracks(self, crate_id: str, track_ids: List[str]) -> bool:
        """Update the track list for a crate."""
        if not self.client:
            return False
        try:
            # Delete existing tracks
            self.client.table("crate_tracks").delete().eq("crate_id", crate_id).execute()
            # Insert new tracks
            for i, tid in enumerate(track_ids):
                self.client.table("crate_tracks").insert({
                    "crate_id": crate_id, "trackid": tid, "position": i,
                }).execute()
            return True
        except Exception as e:
            logger.warning(f"update_crate_tracks failed: {e}")
            return False

    async def delete_crate(self, crate_id: str) -> bool:
        """Delete a crate and its children (cascade)."""
        if not self.client:
            return False
        try:
            # Delete child crates first (recursive)
            children = self.client.table("crates") \
                .select("id").eq("parent_crate_id", crate_id).execute()
            for child in (children.data or []):
                await self.delete_crate(child["id"])
            # Delete tracks and crate
            self.client.table("crate_tracks").delete().eq("crate_id", crate_id).execute()
            self.client.table("crates").delete().eq("id", crate_id).execute()
            return True
        except Exception as e:
            logger.warning(f"delete_crate failed: {e}")
            return False

    # ── Online learning operations ──────────────────────────────────────────

    async def get_recent_corrections(self, since_days: int = 30) -> List[Dict[str, Any]]:
        """Fetch recent tag corrections from the tag_corrections table."""
        if not self.client:
            return []
        try:
            cutoff = (datetime.utcnow() - timedelta(days=since_days)).isoformat()
            response = self.client.table("tag_corrections") \
                .select("*") \
                .gte("created_at", cutoff) \
                .order("created_at", desc=True) \
                .execute()
            return response.data or []
        except Exception as e:
            logger.warning(f"get_recent_corrections failed (table may not exist): {e}")
            return []

    async def get_auto_tagged_tracks(self) -> List[Dict[str, Any]]:
        """Fetch tracks with tag_source='auto' that have embeddings."""
        if not self.client:
            return []
        try:
            # Join tracks (for embedding) with track_labels (for tag_source filter)
            response = self.client.table("tracks") \
                .select("trackid, embedding, track_labels!inner(tag_source)") \
                .not_.is_("embedding", "null") \
                .eq("track_labels.tag_source", "auto") \
                .execute()
            return response.data or []
        except Exception as e:
            logger.warning(f"get_auto_tagged_tracks failed (table may not exist): {e}")
            return []

    async def log_propagation(self, corrections_applied: int, tracks_updated: int) -> None:
        """Insert a row into correction_propagation_log."""
        if not self.client:
            return
        try:
            self.client.table("correction_propagation_log").insert({
                "corrections_applied": corrections_applied,
                "tracks_updated": tracks_updated,
                "propagated_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            logger.warning(f"log_propagation failed (table may not exist): {e}")

    async def get_propagation_status(self) -> Optional[Dict[str, Any]]:
        """Return the most recent propagation log entry and cumulative totals."""
        if not self.client:
            return None
        try:
            response = self.client.table("correction_propagation_log") \
                .select("*") \
                .order("propagated_at", desc=True) \
                .limit(1) \
                .execute()
            rows = response.data or []
            if not rows:
                return {"last_propagation": None, "total_propagated": 0}

            latest = rows[0]

            # Sum total propagated tracks across all runs
            all_resp = self.client.table("correction_propagation_log") \
                .select("tracks_updated") \
                .execute()
            total = sum(r.get("tracks_updated", 0) for r in (all_resp.data or []))

            return {
                "last_propagation": latest.get("propagated_at"),
                "total_propagated": total,
            }
        except Exception as e:
            logger.warning(f"get_propagation_status failed (table may not exist): {e}")
            return None

    # ── Playlist operations ────────────────────────────────────────────────────

    async def create_playlist(
        self, playlist_id: str, name: str, source: str = "rekordbox", source_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new playlist."""
        if not self.client:
            return {"id": playlist_id, "name": name, "source": source}
        try:
            payload = {"id": playlist_id, "name": name, "source": source}
            if source_path:
                payload["source_path"] = source_path
            self.client.table("playlists").insert(payload).execute()
            return {"id": playlist_id, "name": name, "source": source}
        except Exception as e:
            logger.warning(f"create_playlist failed (table may not exist): {e}")
            return {"id": playlist_id, "name": name, "source": source}

    async def get_playlists(self) -> List[Dict[str, Any]]:
        """List all playlists with track counts."""
        if not self.client:
            return []
        try:
            response = self.client.table("playlists") \
                .select("*") \
                .order("created_at", desc=True) \
                .execute()

            playlists = []
            for row in response.data or []:
                # Get track count
                count = 0
                try:
                    count_resp = self.client.table("playlist_tracks") \
                        .select("id", count="exact") \
                        .eq("playlist_id", row["id"]) \
                        .execute()
                    count = count_resp.count if hasattr(count_resp, "count") and count_resp.count else 0
                except Exception:
                    pass

                playlists.append({
                    "id": row["id"],
                    "name": row.get("name", ""),
                    "source": row.get("source"),
                    "track_count": count,
                    "created_at": row.get("created_at"),
                })
            return playlists
        except Exception as e:
            logger.warning(f"get_playlists failed (table may not exist): {e}")
            return []

    async def get_playlist(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """Get a playlist with its tracks (joined with tracks table)."""
        if not self.client:
            return None
        try:
            # Get playlist metadata
            resp = self.client.table("playlists") \
                .select("*") \
                .eq("id", playlist_id) \
                .single() \
                .execute()
            if not resp.data:
                return None

            # Get ordered track IDs
            tracks_resp = self.client.table("playlist_tracks") \
                .select("trackid, position") \
                .eq("playlist_id", playlist_id) \
                .order("position") \
                .execute()

            track_ids = [r["trackid"] for r in (tracks_resp.data or [])]
            tracks = await self.get_tracks_by_ids(track_ids)

            track_dicts = []
            for t in tracks:
                track_dicts.append({
                    "trackid": t.trackid,
                    "title": t.title,
                    "artist": t.artist,
                    "bpm": t.bpm,
                    "key": t.key,
                    "duration": t.duration,
                    "energy": t.energy,
                    "semantic_tags": t.semantic_tags or [],
                    "vibe_descriptors": t.vibe_descriptors or [],
                })

            return {
                "id": playlist_id,
                "name": resp.data.get("name", ""),
                "source": resp.data.get("source"),
                "tracks": track_dicts,
            }
        except Exception as e:
            logger.warning(f"get_playlist failed: {e}")
            return None

    async def update_playlist_tracks(self, playlist_id: str, track_ids: List[str]) -> bool:
        """Replace all tracks in a playlist with a new ordered list."""
        if not self.client:
            return False
        try:
            # Delete existing tracks
            self.client.table("playlist_tracks") \
                .delete() \
                .eq("playlist_id", playlist_id) \
                .execute()
            # Insert new tracks with position
            for i, tid in enumerate(track_ids):
                self.client.table("playlist_tracks").insert({
                    "playlist_id": playlist_id,
                    "trackid": tid,
                    "position": i,
                }).execute()
            return True
        except Exception as e:
            logger.warning(f"update_playlist_tracks failed: {e}")
            return False

    async def delete_playlist(self, playlist_id: str) -> bool:
        """Delete a playlist and its track associations."""
        if not self.client:
            return False
        try:
            # Delete track associations first
            self.client.table("playlist_tracks") \
                .delete() \
                .eq("playlist_id", playlist_id) \
                .execute()
            # Delete playlist
            self.client.table("playlists") \
                .delete() \
                .eq("id", playlist_id) \
                .execute()
            return True
        except Exception as e:
            logger.warning(f"delete_playlist failed: {e}")
            return False

    async def rename_playlist(self, playlist_id: str, new_name: str) -> bool:
        """Update the name of a playlist."""
        if not self.client:
            return False
        try:
            self.client.table("playlists") \
                .update({"name": new_name}) \
                .eq("id", playlist_id) \
                .execute()
            return True
        except Exception as e:
            logger.warning(f"rename_playlist failed (table may not exist): {e}")
            return False

    async def get_playlist_tree(self) -> List[Dict[str, Any]]:
        """Fetch all playlists ordered by created_at with track counts."""
        if not self.client:
            return []
        try:
            response = self.client.table("playlists") \
                .select("*") \
                .order("created_at") \
                .execute()

            playlists = []
            for row in response.data or []:
                # Get track count
                count = 0
                try:
                    count_resp = self.client.table("playlist_tracks") \
                        .select("id", count="exact") \
                        .eq("playlist_id", row["id"]) \
                        .execute()
                    count = count_resp.count if hasattr(count_resp, "count") and count_resp.count else 0
                except Exception:
                    pass

                playlists.append({
                    "id": row["id"],
                    "name": row.get("name", ""),
                    "parent_id": row.get("parent_id"),
                    "description": row.get("description"),
                    "track_count": count,
                    "created_at": row.get("created_at"),
                })
            return playlists
        except Exception as e:
            logger.warning(f"get_playlist_tree failed (table may not exist): {e}")
            return []

    async def get_track_pool(self) -> List[Dict[str, Any]]:
        """Fetch all tracks joined with labels. Lightweight — no embeddings or UMAP coords."""
        if not self.client:
            return []
        try:
            response = self.client.table("tracks") \
                .select("trackid, title, artist, bpm, key, filepath, track_labels(energy, semantic_tags, vibe)") \
                .execute()

            tracks = []
            for row in response.data or []:
                labels = row.get("track_labels", {}) or {}
                tracks.append({
                    "trackid": row["trackid"],
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "bpm": row.get("bpm"),
                    "key": row.get("key"),
                    "filepath": row.get("filepath", ""),
                    "energy": labels.get("energy"),
                    "semantic_tags": labels.get("semantic_tags") or [],
                    "vibe": labels.get("vibe") or [],
                })
            return tracks
        except Exception as e:
            logger.warning(f"get_track_pool failed (table may not exist): {e}")
            return []

    async def add_tracks_to_playlist(self, playlist_id: str, track_ids: List[str]) -> bool:
        """Append tracks to a playlist starting after the current max position."""
        if not self.client:
            return False
        try:
            # Get current max position
            max_pos = -1
            try:
                pos_resp = self.client.table("playlist_tracks") \
                    .select("position") \
                    .eq("playlist_id", playlist_id) \
                    .order("position", desc=True) \
                    .limit(1) \
                    .execute()
                if pos_resp.data:
                    max_pos = pos_resp.data[0]["position"]
            except Exception:
                pass

            # Insert new tracks starting at max_position + 1
            for i, tid in enumerate(track_ids):
                self.client.table("playlist_tracks").insert({
                    "playlist_id": playlist_id,
                    "trackid": tid,
                    "position": max_pos + 1 + i,
                }).execute()
            return True
        except Exception as e:
            logger.warning(f"add_tracks_to_playlist failed (table may not exist): {e}")
            return False

    async def remove_tracks_from_playlist(self, playlist_id: str, track_ids: List[str]) -> bool:
        """Remove specific tracks from a playlist."""
        if not self.client:
            return False
        try:
            for tid in track_ids:
                self.client.table("playlist_tracks") \
                    .delete() \
                    .eq("playlist_id", playlist_id) \
                    .eq("trackid", tid) \
                    .execute()
            return True
        except Exception as e:
            logger.warning(f"remove_tracks_from_playlist failed (table may not exist): {e}")
            return False

    async def close(self):
        """Clean up connections."""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Database connections closed")

# Global instance with enhanced configuration
db_manager = DatabaseManager(enable_caching=True, pool_size=10)