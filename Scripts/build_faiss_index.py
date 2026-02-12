# scripts/build_faiss_index.py
import faiss
import numpy as np
import asyncio
import pickle
from Backend.data.db_interface import DatabaseManager

async def build_faiss_index():
    """Build FAISS index from database embeddings"""
    print("🔨 Building FAISS index...")

    # Initialize database
    await DatabaseManager.initialize()

    # Get all tracks with embeddings
    async with DatabaseManager.pool.acquire() as conn:
        rows = await conn.fetch("SELECT trackid, embedding FROM tracks WHERE embedding IS NOT NULL")

    if not rows:
        print("❌ No embeddings found in database!")
        return

    # Convert to numpy arrays
    track_ids = [row['trackid'] for row in rows]
    embeddings = np.array([row['embedding'] for row in rows], dtype=np.float32)

    print(f"📊 Processing {len(embeddings)} tracks with {embeddings.shape[1]}-dim embeddings")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index and mapping
    faiss.write_index(index, "data/faiss_index.bin")

    with open("data/track_id_mapping.pkl", "wb") as f:
        pickle.dump(track_ids, f)

    print(f"✅ FAISS index built and saved! {index.ntotal} vectors indexed.")

if __name__ == "__main__":
    asyncio.run(build_faiss_index())