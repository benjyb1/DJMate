#!/usr/bin/env python3
"""Compute 3D UMAP coordinates for all tracks with embeddings.

Usage:
    python scripts/3d_coordinator.py
    python scripts/3d_coordinator.py --dry-run
    python scripts/3d_coordinator.py --limit 500
"""
import argparse
import logging
import os
import sys
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute 3D UMAP coordinates")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=0, help="Max tracks (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)

    supabase = create_client(url, key)

    logger.info("Fetching tracks with embeddings...")
    query = supabase.table("tracks").select("trackid, embedding").neq("embedding", None)
    if args.limit > 0:
        query = query.limit(args.limit)
    response = query.execute()
    tracks = response.data or []

    if not tracks:
        logger.warning("No tracks with embeddings found.")
        return

    logger.info(f"Found {len(tracks)} tracks")

    ids, vectors = [], []
    for t in tracks:
        emb = t.get("embedding")
        if emb is None:
            continue
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip("[]").split(",")]
        ids.append(t["trackid"])
        vectors.append(emb)

    if len(vectors) < 5:
        logger.warning(f"Only {len(vectors)} valid embeddings — need >= 5 for UMAP")
        return

    X = np.array(vectors)
    logger.info(f"Embedding matrix: {X.shape}")

    try:
        from umap import UMAP
    except ImportError:
        logger.error("umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    n_neighbors = min(15, len(vectors) - 1)
    logger.info(f"Running UMAP (n_neighbors={n_neighbors})...")
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(X)

    for dim in range(3):
        col = coords[:, dim]
        mn, mx = col.min(), col.max()
        if mx > mn:
            coords[:, dim] = 2 * (col - mn) / (mx - mn) - 1

    if args.dry_run:
        logger.info("Dry run — sample coordinates:")
        for i in range(min(5, len(ids))):
            logger.info(f"  Track {ids[i]}: ({coords[i][0]:.3f}, {coords[i][1]:.3f}, {coords[i][2]:.3f})")
        return

    logger.info("Writing coordinates to database...")
    updated = 0
    for i, tid in enumerate(ids):
        try:
            supabase.table("tracks").update({
                "x_coord": float(coords[i][0]),
                "y_coord": float(coords[i][1]),
                "z_coord": float(coords[i][2]),
            }).eq("trackid", tid).execute()
            updated += 1
        except Exception as e:
            logger.warning(f"Failed to update track {tid}: {e}")
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(ids)}")

    logger.info(f"Done! Updated {updated}/{len(ids)} tracks.")


if __name__ == "__main__":
    main()
