"""
Database interface for DJMate session and track management.
Connects to Supabase with schema:
- tracks: trackid, filepath, title, artist, album, bpm, key, embedding
- track_labels: trackid, semantic_tags, energy, vibe
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


class DatabaseManager:
    """Manages database operations for sessions, tracks, and crates."""

    def __init__(self):
        """Initialize the Supabase database connection."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            print("⚠️  Warning: SUPABASE_URL and SUPABASE_KEY not found in environment")
            print("   Set these in your .env file for database functionality")
            self.client = None
        else:
            self.client: Client = create_client(supabase_url, supabase_key)

    async def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single track with its labels."""
        if not self.client:
            return None

        try:
            response = self.client.table("tracks") \
                .select("*, track_labels(semantic_tags, energy, vibe)") \
                .eq("trackid", track_id) \
                .single() \
                .execute()

            return response.data if response.data else None
        except Exception as e:
            print(f"Error fetching track {track_id}: {e}")
            return None

    async def get_tracks_by_ids(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple tracks by their IDs."""
        if not self.client:
            return []

        try:
            response = self.client.table("tracks") \
                .select("*, track_labels(semantic_tags, energy, vibe)") \
                .in_("trackid", track_ids) \
                .execute()

            return response.data if response.data else []
        except Exception as e:
            print(f"Error fetching tracks: {e}")
            return []

    async def get_track_labels(self, track_id: str) -> Dict[str, Any]:
        """Get labels for a specific track."""
        if not self.client:
            return {}

        try:
            response = self.client.table("track_labels") \
                .select("semantic_tags, energy, vibe") \
                .eq("trackid", track_id) \
                .single() \
                .execute()

            if response.data:
                return {
                    "tags": response.data.get("semantic_tags", []),
                    "energy": response.data.get("energy", 0.5),
                    "vibe_descriptors": response.data.get("vibe", [])
                }
            return {}
        except Exception as e:
            print(f"Error fetching labels for track {track_id}: {e}")
            return {}

    async def get_tracks_by_semantic_filter(
            self,
            structured_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter tracks based on semantic tags, energy, BPM, and other criteria.
        """
        if not self.client:
            return []

        try:
            query = self.client.table("tracks").select(
                "trackid, filepath, title, artist, album, bpm, key, embedding, "
                "track_labels(semantic_tags, energy, vibe)"
            )

            # Apply BPM filter
            bpm_range = structured_query.get("bpm_range")
            if bpm_range:
                query = query.gte("bpm", bpm_range[0]).lte("bpm", bpm_range[1])

            # Execute query
            response = query.execute()

            if not response.data:
                return []

            # Post-filter by tags and energy (since Supabase doesn't support nested filtering easily)
            results = []
            tags = structured_query.get("tags", [])
            energy_range = structured_query.get("energy_range")

            for track in response.data:
                # Check if track has labels
                if not track.get("track_labels"):
                    continue

                labels = track["track_labels"]

                # Filter by tags
                if tags:
                    track_tags = labels.get("semantic_tags", [])
                    if not any(tag in track_tags for tag in tags):
                        continue

                # Filter by energy
                if energy_range:
                    track_energy = labels.get("energy", 0.5)
                    if not (energy_range[0] <= track_energy <= energy_range[1]):
                        continue

                # Flatten the structure
                track_flat = {
                    "track_id": track["trackid"],
                    "filepath": track["filepath"],
                    "title": track["title"],
                    "artist": track["artist"],
                    "album": track["album"],
                    "bpm": track["bpm"],
                    "key": track["key"],
                    "embedding": track["embedding"],
                    "labels": labels
                }
                results.append(track_flat)

            return results

        except Exception as e:
            print(f"Error filtering tracks: {e}")
            return []

    async def get_session_context(
            self,
            session_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve session context for a given session ID.
        TODO: Implement session storage in Supabase
        """
        if not session_id:
            return None

        # Placeholder - implement session table later
        return {
            "session_id": session_id,
            "previous_tracks": [],
            "preferences": {}
        }

    async def update_crate(
            self,
            session_id: str,
            tracks: List[str],
            sequence: List[int],
            validation_results: Dict[str, Any]
    ) -> bool:
        """
        Update or create a crate with validated track sequence.
        TODO: Implement crates table in Supabase
        """
        print(f"Crate updated for session {session_id}")
        print(f"  Tracks: {len(tracks)}")
        print(f"  Sequence score: {validation_results.get('overall_score', 0.0)}")
        return True
    # Add this inside your DatabaseManager class
    async def find_similar_tracks(
            self,
            query_embedding: List[float],
            limit: int = 20,
            threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find tracks with similar embeddings using pgvector RPC."""
        if not self.client:
            return []

        try:
            # We call the 'match_tracks' function we created in SQL
            response = self.client.rpc(
                'match_tracks',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': limit
                }
            ).execute()

            return response.data if response.data else []
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []