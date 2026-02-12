"""
explore_model_outputs.py - Find the correct output node for 1280-dim embeddings

The model has multiple output nodes. We need to find which one gives us 1280-dim embeddings.
"""

import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
import sys

MODEL_PATH = '/Kings Coding Club/DJMate/Models/discogs-effnet-bs64.pb'

if len(sys.argv) < 2:
    print("Usage: python explore_model_outputs.py <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

print("="*80)
print("EXPLORING MODEL OUTPUT NODES")
print("="*80)
print(f"Model: {MODEL_PATH}")
print(f"Audio: {audio_file}")
print()

# Load audio
print("Loading audio...")
loader = MonoLoader(filename=audio_file, sampleRate=16000, resampleQuality=4)
audio = loader()
print(f"Audio: {len(audio)} samples ({len(audio)/16000:.1f}s)")
print()

# List of nodes to try based on your diagnostic output
test_nodes = [
    None,  # Default
    "PartitionedCall",
    "PartitionedCall:0",
    "PartitionedCall:1",
    "StatefulPartitionedCall",
    "StatefulPartitionedCall:0",
    "StatefulPartitionedCall:1",
]

results = []

for node in test_nodes:
    node_name = node if node else "Default (no output specified)"
    print(f"\n{'='*80}")
    print(f"Testing: {node_name}")
    print(f"{'='*80}")

    try:
        # Try to load model with this output
        if node:
            model = TensorflowPredictEffnetDiscogs(
                graphFilename=MODEL_PATH,
                output=node
            )
        else:
            model = TensorflowPredictEffnetDiscogs(
                graphFilename=MODEL_PATH
            )

        # Compute embedding
        embedding = model(audio)
        embedding = np.array(embedding)

        print(f"✓ SUCCESS")
        print(f"  Shape: {embedding.shape}")
        print(f"  Size: {embedding.size}")
        print(f"  Dtype: {embedding.dtype}")

        # Check if we have 1280 dimensions
        if 1280 in embedding.shape:
            print(f"  *** CONTAINS 1280 DIMENSIONS! ***")

            # If it's 2D, show which axis has 1280
            if embedding.ndim == 2:
                if embedding.shape[1] == 1280:
                    print(f"  → Shape is ({embedding.shape[0]}, 1280) - AVERAGE AXIS 0")
                    avg = np.mean(embedding, axis=0)
                    print(f"  → Averaged shape: {avg.shape}")
                elif embedding.shape[0] == 1280:
                    print(f"  → Shape is (1280, {embedding.shape[1]}) - AVERAGE AXIS 1")
                    avg = np.mean(embedding, axis=1)
                    print(f"  → Averaged shape: {avg.shape}")

        results.append({
            'node': node_name,
            'success': True,
            'shape': embedding.shape,
            'size': embedding.size
        })

    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append({
            'node': node_name,
            'success': False,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY OF ALL TESTS")
print("="*80)

for r in results:
    if r['success']:
        has_1280 = 1280 in r['shape'] if isinstance(r['shape'], tuple) else r['size'] == 1280
        marker = "⭐" if has_1280 else "  "
        print(f"{marker} {r['node']:40s} → {r['shape']}")
    else:
        print(f"  {r['node']:40s} → FAILED")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find the best option
best = None
for r in results:
    if r['success']:
        if isinstance(r['shape'], tuple) and 1280 in r['shape']:
            best = r
            break

if best:
    print(f"✓ Use output node: '{best['node']}'")
    print(f"  This gives shape: {best['shape']}")
    if best['shape'][1] == 1280:
        print(f"  Average with: np.mean(embeddings, axis=0)")
else:
    print("✗ No node found with 1280 dimensions")
    print("  Your model may actually output 400-dim embeddings")
    print("  Check the model documentation or source")