"""Convert Dreamer npy evaluation results to benchmark JSON format."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_results import load_eval_run


def convert_and_save(rollout_dir: Path, output_path: Path):
    """Convert npy rollout data to JSON and save."""
    result = load_eval_run(rollout_dir, method_name='dreamerv3')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(result['trials'])} trials to {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent
    results_base = base.parent / "results"

    # 2dot
    convert_and_save(
        base / "dreamer_rollouts/2dot_12m",
        results_base / "final_2dot/dreamerv3_2dots.json",
    )

    # 4dot
    convert_and_save(
        base / "dreamer_rollouts/4dot_12m",
        results_base / "final_4dot/dreamerv3_4dots.json",
    )
