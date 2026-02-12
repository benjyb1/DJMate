
"""
TrackEmbedder.py - Auto-detecting Frame Averaging
Feeds audio to EffNet model and averages per-frame embeddings into one summary vector
"""

import numpy as np
from pathlib import Path
import logging
from typing import Optional
from essentia.standard import MonoLoader, TensorflowPredict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------ Configuration ------------------
MODEL_PATH = '/Kings Coding Club/DJMate/Models/discogs-effnet-bs64.pb'

# Global model instance (loaded once, reused for all tracks)
EMBEDDING_MODEL = None


def load_model():
    """Load Discogs-EffNet model once globally"""
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        try:
            logger.info("Loading Discogs-EffNet model...")
            EMBEDDING_MODEL = TensorflowPredict(
                graphFilename=MODEL_PATH,
                output='PartitionedCall:1'
            )
            logger.info("✓ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    return EMBEDDING_MODEL


# Load model on module import
try:
    load_model()
except Exception as e:
    logger.error(f"Could not initialize embedding model: {e}")
    EMBEDDING_MODEL = None


def compute_embedding(filepath: str, strategy: str = 'full') -> Optional[np.ndarray]:
    """
    Compute embedding by averaging per-frame embeddings

    Process:
    1. Load full audio track at 16kHz
    2. Model outputs (num_frames, embedding_dim)
    3. Average across all frames: mean(axis=0) → (embedding_dim,)
    4. Returns single summary vector for the entire track

    Args:
        filepath: Path to audio file
        strategy: 'full' (only option, kept for compatibility)

    Returns:
        numpy array of shape (embedding_dim,) or None on failure
    """
    if EMBEDDING_MODEL is None:
        logger.error("Embedding model not loaded")
        return None

    try:
        # Load audio at 16kHz (model requirement)
        logger.info(f"Loading audio from {filepath}...")
        loader = MonoLoader(
            filename=filepath,
            sampleRate=16000,
            resampleQuality=4
        )
        audio = loader()

        total_duration = len(audio) / 16000
        logger.info(f"Audio: {len(audio)} samples ({total_duration:.1f}s)")

        # Validate minimum audio length
        if len(audio) < 16000:  # Less than 1 second
            logger.warning(f"Audio too short ({len(audio)} samples = {len(audio) / 16000:.2f}s)")
            return None

        # Compute embeddings
        logger.info(f"Computing per-frame embeddings...")
        embeddings = EMBEDDING_MODEL(audio)

        # Convert to numpy array
        embeddings = np.array(embeddings)

        logger.info(f"Raw output shape: {embeddings.shape}")

        # Validate we got a 2D array (num_frames, embedding_dim)
        if embeddings.ndim != 2:
            logger.error(f"Expected 2D output, got {embeddings.ndim}D: {embeddings.shape}")
            return None

        num_frames, embedding_dim = embeddings.shape
        logger.info(f"Model output: {num_frames} frames × {embedding_dim} dimensions")

        # Mean + Std pooling across frames
        logger.info(f"Computing mean + std pooling over {num_frames} frames...")

        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)

        # Concatenate mean and std
        final_embedding = np.concatenate([mean_embedding, std_embedding], axis=0)
        final_embedding = final_embedding.flatten()

        logger.info(f"✓ Final embedding shape: {final_embedding.shape}")
        logger.info(
            f"✓ Stats - Min: {final_embedding.min():.4f}, "
            f"Max: {final_embedding.max():.4f}, "
            f"Mean: {final_embedding.mean():.4f}"
        )

        return final_embedding

    except Exception as e:
        logger.error(f"Failed to compute embedding for {filepath}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def validate_embedding(embedding: Optional[np.ndarray]) -> bool:
    """
    Validate embedding before upload to Supabase

    Args:
        embedding: numpy array or None

    Returns:
        True if valid, False otherwise
    """
    if embedding is None:
        logger.error("Embedding is None")
        return False

    if not isinstance(embedding, np.ndarray):
        logger.error(f"Embedding is not numpy array: {type(embedding)}")
        return False

    if embedding.size == 0:
        logger.error("Embedding is empty (size 0)")
        return False

    if embedding.ndim != 1:
        logger.error(f"Embedding should be 1D, got {embedding.ndim}D: {embedding.shape}")
        return False

    if embedding.size != 2560:
        logger.error(f"Embedding has wrong size: {embedding.size}, expected 2560")
        return False

    if np.isnan(embedding).any():
        logger.error("Embedding contains NaN values")
        return False

    if np.isinf(embedding).any():
        logger.error("Embedding contains Inf values")
        return False

    # Check if embedding is all zeros (suspicious)
    if np.all(embedding == 0):
        logger.warning("Embedding is all zeros (may be invalid)")
        return False

    logger.info(f"✓ Embedding validation passed (dimension: {embedding.size})")
    return True


# ------------------ Testing Functions ------------------

def test_single_track(filepath: str):
    """
    Test embedding computation on a single track

    Args:
        filepath: Path to test audio file
    """
    print("\n" + "=" * 60)
    print(f"Testing embedding computation on: {Path(filepath).name}")
    print("=" * 60)

    emb = compute_embedding(filepath, strategy='full')

    if emb is not None:
        print(f"\n✓ Embedding computed successfully")
        print(f"  Shape: {emb.shape}")
        print(f"  Size: {emb.size}")
        print(f"  Min: {emb.min():.4f}")
        print(f"  Max: {emb.max():.4f}")
        print(f"  Mean: {emb.mean():.4f}")
        print(f"  Std: {emb.std():.4f}")

        if validate_embedding(emb):
            print("\n✓ Embedding validation PASSED")
            print("✓ Ready for Supabase upload")
        else:
            print("\n✗ Embedding validation FAILED")
    else:
        print("\n✗ Failed to compute embedding")

    print("\n" + "=" * 60)


# ------------------ CLI Interface ------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("TrackEmbedder - Compute Discogs-EffNet embeddings (frame-averaged)")
        print("\nUsage:")
        print("  python TrackEmbedder.py <audio_file>")
        print("\nExample:")
        print("  python TrackEmbedder.py track.mp3")
        sys.exit(1)

    test_file = sys.argv[1]
    if not Path(test_file).exists():
        print(f"Error: File not found: {test_file}")
        sys.exit(1)

    test_single_track(test_file)