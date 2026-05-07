"""Merge per-GPU shard npzs from eval_multi_N.py into a single staircase_scan_N{N}.npz."""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard .npz paths")
    ap.add_argument("--out", required=True, help="merged output .npz path")
    args = ap.parse_args()

    shards = [np.load(p, allow_pickle=True) for p in args.shards]
    # Sort by seed_offset so the merged arrays are in seed order.
    shards = sorted(shards, key=lambda d: int(d["seed_offset"]))

    keys_to_concat = [
        "reward_greedy", "reward_random",
        "staircase_greedy", "staircase_random",
    ]
    merged = {}
    for k in keys_to_concat:
        merged[k] = np.concatenate([s[k] for s in shards], axis=0)

    n_qubits = int(shards[0]["n_qubits"])
    for s in shards:
        assert int(s["n_qubits"]) == n_qubits, "n_qubits mismatch across shards"

    merged["n_seeds"] = sum(int(s["n_seeds"]) for s in shards)
    merged["n_qubits"] = n_qubits
    merged["n_allxy"] = int(shards[0]["n_allxy"])
    merged["n_steps"] = int(shards[0]["n_steps"])
    merged["checkpoint"] = str(shards[0]["checkpoint"])
    merged["param_config"] = str(shards[0]["param_config"])

    np.savez(args.out, **merged)
    print(f"Merged {len(shards)} shards → {args.out}")
    print(f"  n_seeds total: {merged['n_seeds']}  n_qubits: {merged['n_qubits']}")
    print(f"  reward_greedy {merged['reward_greedy'].shape}")
    print(f"  staircase_greedy {merged['staircase_greedy'].shape}")
    print(f"  greedy final mean: {merged['reward_greedy'][:, -1].mean():.3f}")
    print(f"  random final mean: {merged['reward_random'][:, -1].mean():.3f}")


if __name__ == "__main__":
    main()
