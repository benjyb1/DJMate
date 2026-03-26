"""
Suggested playlist generator — seed-and-expand clustering on Discogs EffNet embeddings.

Produces ~10 playlists of 7-21 tracks grouped by sonic similarity,
with LLM-generated 2-word names.
"""

import os
import random
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from statistics import median
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# ── Tunable parameters ────────────────────────────────────────────────────────
MIN_SEED_DISTANCE = 0.4       # cosine distance between seeds
SEED_DISTANCE_FLOOR = 0.25    # minimum after retries
EXPANSION_RADIUS = 0.28       # max cosine distance from seed to include
BPM_DEVIATION_MAX = 15        # normalised BPM
BPM_CANONICAL_MIN = 80
BPM_CANONICAL_MAX = 160
TARGET_PLAYLIST_COUNT = 10
MIN_PLAYLIST_SIZE = 7
MAX_PLAYLIST_SIZE = 21


def _normalise_bpm(bpm: float) -> float:
    """Fold BPM into 80-160 canonical range by halving or doubling."""
    if bpm is None or bpm <= 0:
        return 120.0  # fallback
    while bpm > BPM_CANONICAL_MAX:
        bpm /= 2
    while bpm < BPM_CANONICAL_MIN:
        bpm *= 2
    return bpm


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance for normalised embeddings: 1 - dot."""
    # embeddings should be L2-normalised already
    sim = embeddings @ embeddings.T
    return 1.0 - sim


def generate_playlists(
    track_ids: List[int],
    embeddings: np.ndarray,
    bpms: List[float],
) -> List[Dict[str, Any]]:
    """
    Core algorithm: seed-and-expand with minimum seed distance + Voronoi assignment.

    Parameters
    ----------
    track_ids : list of track IDs matching embedding rows
    embeddings : (N, D) numpy array, L2-normalised
    bpms : list of BPM values matching track_ids

    Returns
    -------
    list of dicts: [{"seed_trackid": int, "track_ids": [int, ...]}]
    """
    n = len(track_ids)
    if n < MIN_PLAYLIST_SIZE:
        logger.warning(f"Only {n} tracks — too few for playlist generation")
        return []

    # Build index maps
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    norm_bpms = [_normalise_bpm(b) for b in bpms]

    # ── Seed selection ────────────────────────────────────────────────────
    seed_indices = []
    available = set(range(n))
    min_dist = MIN_SEED_DISTANCE

    while len(seed_indices) < TARGET_PLAYLIST_COUNT and available:
        candidates = []
        for idx in available:
            if not seed_indices:
                candidates.append(idx)
                continue
            # Check distance to all existing seeds
            seed_embs = embeddings[seed_indices]  # (num_seeds, D)
            dists = 1.0 - (seed_embs @ embeddings[idx])  # cosine distances
            if np.all(dists > min_dist):
                candidates.append(idx)

        if not candidates:
            min_dist -= 0.05
            if min_dist < SEED_DISTANCE_FLOOR:
                break
            continue

        seed = random.choice(candidates)
        seed_indices.append(seed)
        available.discard(seed)

    if not seed_indices:
        logger.warning("No seeds selected — library may be too small or homogeneous")
        return []

    logger.info(f"Selected {len(seed_indices)} seeds (min_dist={min_dist:.2f})")

    # ── Voronoi assignment: each track → nearest seed ─────────────────────
    seed_embs = embeddings[seed_indices]  # (num_seeds, D)
    all_sims = embeddings @ seed_embs.T   # (N, num_seeds) — cosine similarity
    all_dists = 1.0 - all_sims            # cosine distance

    nearest_seed = np.argmin(all_dists, axis=1)  # index into seed_indices

    # Group tracks by their nearest seed
    cells: Dict[int, List[int]] = {i: [] for i in range(len(seed_indices))}
    for track_idx in range(n):
        cells[nearest_seed[track_idx]].append(track_idx)

    # ── Filter and build playlists ────────────────────────────────────────
    playlists = []
    for seed_pos, seed_idx in enumerate(seed_indices):
        cell = cells[seed_pos]

        # Radius filter — only tracks close enough to the seed
        cell_filtered = []
        for idx in cell:
            if idx == seed_idx:
                continue
            dist = all_dists[idx, seed_pos]
            if dist <= EXPANSION_RADIUS:
                cell_filtered.append((idx, float(dist)))

        # BPM gate (normalised)
        all_cell_bpms = [norm_bpms[seed_idx]] + [norm_bpms[idx] for idx, _ in cell_filtered]
        if not all_cell_bpms:
            continue
        med_bpm = median(all_cell_bpms)
        cell_filtered = [
            (idx, dist) for idx, dist in cell_filtered
            if abs(norm_bpms[idx] - med_bpm) <= BPM_DEVIATION_MAX
        ]

        # Size check (need at least MIN_PLAYLIST_SIZE - 1 neighbours + seed)
        if len(cell_filtered) + 1 < MIN_PLAYLIST_SIZE:
            continue

        # Sort by distance, cap at MAX_PLAYLIST_SIZE - 1 (plus seed)
        cell_filtered.sort(key=lambda x: x[1])
        cell_filtered = cell_filtered[:MAX_PLAYLIST_SIZE - 1]

        playlist_track_ids = [track_ids[seed_idx]] + [track_ids[idx] for idx, _ in cell_filtered]
        playlists.append({
            "seed_trackid": track_ids[seed_idx],
            "track_ids": playlist_track_ids,
        })

    logger.info(f"Generated {len(playlists)} playlists from {n} tracks")
    return playlists


async def name_playlist_with_llm(
    tracks_data: List[Dict[str, Any]],
    providers: List[Dict[str, Any]],
) -> str:
    """
    Generate a 2-word playlist name from aggregated track data.
    Falls back to a simple template if LLM fails.
    """
    # Aggregate tags, vibes, energy, bpm
    all_tags = []
    all_vibes = []
    energies = []
    bpms = []

    for t in tracks_data:
        tags = t.get("semantic_tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        all_tags.extend(tags)

        vibes = t.get("vibe") or t.get("vibes") or []
        if isinstance(vibes, str):
            try:
                vibes = json.loads(vibes)
            except (json.JSONDecodeError, TypeError):
                vibes = []
        all_vibes.extend(vibes)

        if t.get("energy") is not None:
            energies.append(float(t["energy"]))
        if t.get("bpm") is not None:
            bpms.append(float(t["bpm"]))

    top_tags = [tag for tag, _ in Counter(all_tags).most_common(5)]
    top_vibes = [v for v, _ in Counter(all_vibes).most_common(5)]
    avg_energy = round(sum(energies) / len(energies), 1) if energies else 5.0
    bpm_range = f"{min(bpms):.0f}-{max(bpms):.0f}" if bpms else "unknown"

    prompt = (
        f"You are naming DJ playlists. Given tracks with tags {top_tags}, "
        f"vibes {top_vibes}, average energy {avg_energy}/10, BPM range {bpm_range}, "
        f"generate a short playlist name. Maximum 2 words. Be creative and evocative, "
        f"not generic. Examples: 'Velvet Groove', 'Acid Cathedral', 'Neon Drift'. "
        f"Reply with ONLY the name, nothing else."
    )

    # Try each provider
    from openai import AsyncOpenAI
    for provider in providers:
        try:
            client: AsyncOpenAI = provider["client"]
            resp = await client.chat.completions.create(
                model=provider["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.9,
            )
            name = resp.choices[0].message.content.strip().strip('"\'')
            # Validate: max 2 words, not empty
            words = name.split()
            if 1 <= len(words) <= 3 and len(name) < 40:
                return name
        except Exception as e:
            logger.warning(f"LLM naming failed with {provider['name']}: {e}")
            continue

    # Fallback: top tag + energy descriptor
    energy_word = "Burner" if avg_energy > 7 else "Cruise" if avg_energy > 4 else "Ambient"
    tag_word = top_tags[0].title() if top_tags else "Mixed"
    return f"{tag_word} {energy_word}"
