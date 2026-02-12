import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class DatabaseManager:
    def __init__(self):
        self.client = supabase

    async def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single track and its labels"""
        response = self.client.table("tracks") \
            .select(
            "*, track_labels(semantic_tags, energy, vibe)"
        ) \
            .eq("trackid", track_id) \
            .single().execute()

        if response.error:
            return None
        return response.data

    async def get_tracks_by_semantic_filter(
            self,
            tags: List[str] = None,
            energy_range: tuple = None,
            bpm_range: tuple = None
    ) -> List[Dict[str, Any]]:
        """Filter tracks based on semantic tags, energy, and BPM"""
        query = self.client.table("tracks").select("*, track_labels(semantic_tags, energy, vibe)")

        if tags:
            query = query.contains("track_labels.semantic_tags", tags)

        if energy_range:
            query = query.gte("track_labels.energy", energy_range[0]).lte("track_labels.energy", energy_range[1])

        if bpm_range:
            query = query.gte("bpm", bpm_range[0]).lte("bpm", bpm_range[1])

        response = query.execute()
        if response.error:
            return []
        return response.data


db_manager = DatabaseManager()