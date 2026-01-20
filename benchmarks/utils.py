"""
Shared utilities for benchmark results saving, loading, and analysis.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class TrialResult:
    """Result from a single optimization trial."""
    trial_idx: int
    seed: int
    success: bool
    num_iterations: int
    num_function_evals: int
    num_scans: int  # Normalized scan count for fair comparison
    final_objective: float
    final_plunger_distances: List[float]
    final_barrier_distances: List[float]
    convergence_history: List[float]  # Local objective at end of each sweep
    global_objective_history: List[float] = field(default_factory=list)  # Global objective at each func eval
    voltage_history: List[List[float]] = field(default_factory=list)  # Voltages at each func eval (optional)


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a method."""
    method: str
    mode: str  # "joint" or "pairwise"
    num_dots: int
    use_barriers: bool
    num_trials: int
    max_iterations: int
    success_threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Pairwise mode parameters (optional)
    max_sweeps: Optional[int] = None
    cap_per_plunger: Optional[float] = None
    cap_per_barrier: Optional[float] = None
    threshold_per_plunger: Optional[float] = None
    threshold_per_barrier: Optional[float] = None
    simplex_step_plunger: Optional[float] = None
    simplex_step_barrier: Optional[float] = None
    xatol: Optional[float] = None
    fatol: Optional[float] = None

    # Aggregated results
    success_rate: float = 0.0
    mean_iterations: float = 0.0
    mean_function_evals: float = 0.0
    mean_scans: float = 0.0

    # Individual trial results
    trials: List[TrialResult] = field(default_factory=list)

    def compute_stats(self):
        """Compute aggregate statistics from trials."""
        if not self.trials:
            return

        successes = [t for t in self.trials if t.success]
        self.success_rate = len(successes) / len(self.trials)
        self.mean_iterations = np.mean([t.num_iterations for t in self.trials])
        self.mean_function_evals = np.mean([t.num_function_evals for t in self.trials])
        self.mean_scans = np.mean([t.num_scans for t in self.trials])


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results(result: BenchmarkResult, path: str = None) -> str:
    """
    Save benchmark results to JSON file.

    Args:
        result: BenchmarkResult instance
        path: Optional custom path. If None, auto-generates in results/

    Returns:
        Path to saved file
    """
    if path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        filename = f"{result.method}_{result.mode}_{result.num_dots}dots_{result.timestamp.replace(':', '-')}.json"
        path = results_dir / filename

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = {
        "method": result.method,
        "mode": result.mode,
        "num_dots": result.num_dots,
        "use_barriers": result.use_barriers,
        "num_trials": result.num_trials,
        "max_iterations": result.max_iterations,
        "success_threshold": result.success_threshold,
        "timestamp": result.timestamp,
        "success_rate": result.success_rate,
        "mean_iterations": result.mean_iterations,
        "mean_function_evals": result.mean_function_evals,
        "mean_scans": result.mean_scans,
        "trials": [asdict(t) for t in result.trials],
    }

    # Add pairwise mode parameters if set
    if result.max_sweeps is not None:
        data["max_sweeps"] = result.max_sweeps
    if result.cap_per_plunger is not None:
        data["cap_per_plunger"] = result.cap_per_plunger
    if result.cap_per_barrier is not None:
        data["cap_per_barrier"] = result.cap_per_barrier
    if result.threshold_per_plunger is not None:
        data["threshold_per_plunger"] = result.threshold_per_plunger
    if result.threshold_per_barrier is not None:
        data["threshold_per_barrier"] = result.threshold_per_barrier
    if result.simplex_step_plunger is not None:
        data["simplex_step_plunger"] = result.simplex_step_plunger
    if result.simplex_step_barrier is not None:
        data["simplex_step_barrier"] = result.simplex_step_barrier
    if result.xatol is not None:
        data["xatol"] = result.xatol
    if result.fatol is not None:
        data["fatol"] = result.fatol

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    return str(path)


def load_results(path: str) -> BenchmarkResult:
    """
    Load benchmark results from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        BenchmarkResult instance
    """
    with open(path) as f:
        data = json.load(f)

    trials = [TrialResult(**t) for t in data.pop("trials", [])]

    result = BenchmarkResult(**data)
    result.trials = trials

    return result


def print_summary(result: BenchmarkResult):
    """Print a summary of benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark Results: {result.method} ({result.mode})")
    print(f"{'=' * 60}")
    print(f"Dots: {result.num_dots}, Barriers: {result.use_barriers}")
    print(f"Trials: {result.num_trials}, Max iterations: {result.max_iterations}")
    print(f"Success threshold: {result.success_threshold}V")
    print(f"-" * 60)
    print(f"Success rate: {result.success_rate * 100:.1f}%")
    print(f"Mean scans: {result.mean_scans:.1f}")

    if result.trials:
        final_objs = [t.final_objective for t in result.trials]
        print(f"Final objective: {np.mean(final_objs):.4f} +/- {np.std(final_objs):.4f}")

        successful_trials = [t for t in result.trials if t.success]
        if successful_trials:
            success_scans = [t.num_scans for t in successful_trials]
            print(f"Scans (successful): {np.mean(success_scans):.1f} +/- {np.std(success_scans):.1f}")

    print(f"{'=' * 60}\n")


def list_results(results_dir: str = None) -> List[str]:
    """List all result files in results directory."""
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    else:
        results_dir = Path(results_dir)

    if not results_dir.exists():
        return []

    return sorted([str(p) for p in results_dir.glob("*.json")])


if __name__ == "__main__":
    # Quick test
    result = BenchmarkResult(
        method="test",
        mode="joint",
        num_dots=2,
        use_barriers=True,
        num_trials=5,
        max_iterations=100,
        success_threshold=0.5,
    )

    # Add some fake trials
    for i in range(5):
        trial = TrialResult(
            trial_idx=i,
            seed=42 + i,
            success=i % 2 == 0,
            num_iterations=50 + i * 10,
            num_function_evals=100 + i * 20,
            num_scans=100 + i * 20,
            final_objective=0.1 + i * 0.05,
            final_plunger_distances=[0.2, 0.3],
            final_barrier_distances=[0.1],
            convergence_history=[1.0, 0.5, 0.2, 0.1],
        )
        result.trials.append(trial)

    result.compute_stats()
    print_summary(result)

    # Test save/load
    path = save_results(result, "/tmp/test_benchmark_result.json")
    print(f"Saved to: {path}")

    loaded = load_results(path)
    print(f"Loaded {len(loaded.trials)} trials")
