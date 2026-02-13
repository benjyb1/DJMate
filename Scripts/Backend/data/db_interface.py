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
            tags = set(structured_query.get("tags", []))
            vibe_descriptors = set(structured_query.get("vibe_descriptors", []))

            for track in response.data:
                labels = track.get("track_labels", {})
                if not labels:
                    continue

                # Tag filtering
                if tags:
                    track_tags = set(labels.get("semantic_tags", []))
                    if not tags.intersection(track_tags):
                        continue

                # Vibe filtering
                if vibe_descriptors:
                    track_vibes = set(labels.get("vibe", []))
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

    def _structure_track_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure raw database response into TrackMetadata format."""
        labels = raw_data.get("track_labels", {})
        if isinstance(labels, list) and labels:
            labels = labels[0]  # Take first label if multiple

        return {
            "trackid": raw_data["trackid"],
            "filepath": raw_data["filepath"],
            "title": raw_data["title"],
            "artist": raw_data["artist"],
            "album": raw_data.get("album"),
            "bpm": raw_data.get("bpm"),
            "key": raw_data.get("key"),
            "duration": raw_data.get("duration"),
            "embedding": raw_data.get("embedding"),
            "semantic_tags": labels.get("semantic_tags", []),
            "energy": labels.get("energy"),
            "vibe_descriptors": labels.get("vibe", [])
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

    async def close(self):
        """Clean up connections."""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Database connections closed")

# Global instance with enhanced configuration
db_manager = DatabaseManager(enable_caching=True, pool_size=10)