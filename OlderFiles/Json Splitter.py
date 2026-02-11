"""
Utility: Split Analysis JSON for Parallel Processing
Splits large analysis_results.json into chunks for parallel Step 2 execution
"""

import json
import sys
from pathlib import Path


def split_analysis_json(input_file: str, num_chunks: int):
    """
    Split analysis JSON into multiple chunks

    Args:
        input_file: Path to analysis_results.json
        num_chunks: Number of chunks to create
    """
    print(f"Loading {input_file}...")

    with open(input_file, 'r') as f:
        data = json.load(f)

    tracks = data['tracks']
    total_tracks = len(tracks)

    print(f"Total tracks: {total_tracks}")
    print(f"Splitting into {num_chunks} chunks...")

    # Calculate chunk size
    chunk_size = (total_tracks + num_chunks - 1) // num_chunks  # Ceiling division

    output_files = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_tracks)

        chunk_tracks = tracks[start_idx:end_idx]

        if not chunk_tracks:
            break

        # Create chunk data
        chunk_data = {
            'stats': {
                'total': len(chunk_tracks),
                'analyzed': len(chunk_tracks),
                'failed': 0
            },
            'tracks': chunk_tracks
        }

        # Generate output filename
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_chunk{i + 1}{input_path.suffix}"

        # Save chunk
        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)

        output_files.append(output_file)
        print(f"  ✓ Created {output_file} ({len(chunk_tracks)} tracks)")

    print(f"\n✅ Split complete! Created {len(output_files)} files")
    print("\nTo run in parallel:")
    print("-" * 60)
    for output_file in output_files:
        print(f"python step2_embed_and_upload.py {output_file} &")
    print("-" * 60)
    print("\nOr run sequentially:")
    print("-" * 60)
    for output_file in output_files:
        print(f"python step2_embed_and_upload.py {output_file}")
    print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python split_json.py <input_file> <num_chunks>")
        print("\nExample:")
        print("  python split_json.py analysis_results.json 4")
        print("\nThis creates 4 smaller JSON files that can be processed in parallel")
        sys.exit(1)

    input_file = sys.argv[1]
    num_chunks = int(sys.argv[2])

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    if num_chunks < 1:
        print("Error: num_chunks must be >= 1")
        sys.exit(1)

    split_analysis_json(input_file, num_chunks)