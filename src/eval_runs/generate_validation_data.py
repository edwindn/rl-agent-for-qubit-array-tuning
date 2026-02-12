#!/usr/bin/env python3
"""
Generate validation data for capacitance model using the same generator as training.
This ensures scans are in the same format the model was trained on (raw, not virtualized).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swarm.qarray_dataset.symmetric_capacitance_generator import GenerationConfig, generate_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate validation data for capacitance model')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--num-dots', type=int, default=4, help='Number of dots')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='./validation_dataset', help='Output directory')
    args = parser.parse_args()

    config = GenerationConfig(
        total_samples=args.num_samples,
        workers=args.workers,
        num_dots=args.num_dots,
        use_barriers=True,
        output_dir=args.output_dir,
        batch_size=100,  # Smaller batches for validation
        seed_base=12345,  # Different seed from training
    )

    print(f"Generating {args.num_samples} validation samples with {args.num_dots} dots...")
    print(f"Output directory: {args.output_dir}")

    generate_dataset(config)

    print("Done!")
