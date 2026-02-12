# src/data/db_interface.py
import asyncpg
import json
import numpy as np
from typing import List, Dict, Any, Optional
import os

class DatabaseManager:
    def __init__(self):
        self.connection_string = os.getenv("DATABASE_URL")
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(self.connection_string)

    async def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single track with labels"""
        async with self.pool.acquire() as conn:
            # Join tracks with track_labels
            query = """
            SELECT t.*, tl.semantic_tags, tl.energy, tl.vibe
            FROM tracks t
            LEFT JOIN track_labels tl ON t.trackid = tl.trackid
            WHERE t.trackid = $1
            """
            row = await conn.fetchrow(query, track_id)
            return dict(row) if row else None

    async def get_tracks_by_semantic_filter(self,
                                            tags: List[str] = None,
                                            energy_range: tuple = None,
                                            bpm_range: tuple = None) -> List[Dict[str, Any]]:
        """Filter tracks by semantic criteria"""
        conditions = []
        params = []
        param_count = 0

        base_query = """
        SELECT t.*, tl.semantic_tags, tl.energy, tl.vibe
        FROM tracks t
        LEFT JOIN track_labels tl ON t.trackid = tl.trackid
        WHERE 1=1
        """

        if tags:
            param_count += 1
            conditions.append(f"tl.semantic_tags ?| ${param_count}")
            params.append(tags)

        if energy_range:
            param_count += 1
            conditions.append(f"tl.energy >= ${param_count}")
            params.append(energy_range[0])
            param_count += 1
            conditions.append(f"tl.energy <= ${param_count}")
            params.append(energy_range[1])

        if bmp_range:
            param_count += 1
            conditions.append(f"t.bpm >= ${param_count}")
            params.append(bpm_range[0])
            param_count += 1
            conditions.append(f"t.bpm <= ${param_count}")
            params.append(bpm_range[1])

        final_query = base_query + " AND " + " AND ".join(conditions) if conditions else base_query

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(final_query, *params)
            return [dict(row) for row in rows]

# Initialize global instance
db_manager = DatabaseManager()