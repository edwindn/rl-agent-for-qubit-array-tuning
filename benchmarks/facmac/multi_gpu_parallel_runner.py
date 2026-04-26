"""
MultiGPUParallelRunner — parallel env rollouts with each worker pinned to a
separate GPU via CUDA_VISIBLE_DEVICES.

Why it exists:
  Vendor's ParallelRunner uses multiprocessing fork, which inherits the parent's
  JAX CUDA binding. Even with `spawn`, the naïve approach of wrapping the env
  constructor in CloudpickleWrapper fails silently: multiprocessing unpickles
  Process args BEFORE the target function runs, and unpickling a partial that
  references PyMARLEnvWrapper triggers `import env_wrapper` -> `swarm` -> `jax`,
  which binds JAX to whatever GPU is visible at that moment (often GPU 0),
  before we can set CUDA_VISIBLE_DEVICES for the worker.

  We sidestep this by passing the args as raw bytes. multiprocessing treats
  bytes as opaque payload -- no classes reimported at arg-passing. The worker
  sets CUDA_VISIBLE_DEVICES FIRST, fixes sys.path, then decodes the bytes and
  does the first env_wrapper import. JAX then binds to the pinned GPU.

Config fields consumed:
  runner:            "multi_gpu_parallel"
  batch_size_run:    N
  worker_gpu_ids:    [1, 2, 3, ...]    # physical GPU id per worker, len == N
"""

from __future__ import annotations

import multiprocessing as mp
import pickle

# Safe module-level imports: nothing that triggers jax/swarm.
from runners.parallel_runner import ParallelRunner


def _pinned_env_worker(gpu_id: int, remote, env_args_dict: dict, args_bytes: bytes):
    """
    Subprocess entry point. Sets CUDA_VISIBLE_DEVICES + JAX flags, re-establishes
    sys.path (spawn children don't inherit in-process sys.path edits), then
    constructs PyMARLEnvWrapper and runs vendor's env-loop body.

    Must not import env_wrapper / swarm / jax before env vars are set.
    """
    import os
    import sys
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

    bench_dir = Path(__file__).resolve().parent
    for p in (bench_dir / "vendor", bench_dir.parent.parent / "src", bench_dir):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    # Now safe to unpickle (may traverse torch, which is fine) and to import
    # env_wrapper (triggers jax import at controlled time, after env vars set).
    args = pickle.loads(args_bytes)
    from env_wrapper import PyMARLEnvWrapper

    env = PyMARLEnvWrapper(env_args=env_args_dict, args=args)

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            reward, terminated, env_info = env.step(data)
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
                "reward": reward,
                "terminated": terminated,
                "info": env_info,
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError(f"Unknown env-worker command: {cmd}")


class MultiGPUParallelRunner(ParallelRunner):

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        worker_gpu_ids = list(getattr(args, "worker_gpu_ids", []))
        if len(worker_gpu_ids) != self.batch_size:
            raise ValueError(
                f"worker_gpu_ids must have length {self.batch_size} (batch_size_run), "
                f"got {len(worker_gpu_ids)}: {worker_gpu_ids}"
            )

        ctx = mp.get_context("spawn")
        self.parent_conns, self.worker_conns = zip(*[ctx.Pipe() for _ in range(self.batch_size)])

        # Pickle args ONCE as raw bytes. Passing bytes through multiprocessing
        # avoids any eager unpickling in the child before it sets env vars.
        import cloudpickle
        args_bytes = cloudpickle.dumps(self.args)
        env_args_dict = dict(self.args.env_args)

        self.ps = [
            ctx.Process(
                target=_pinned_env_worker,
                args=(gpu_id, worker_conn, env_args_dict, args_bytes),
                daemon=True,
            )
            for gpu_id, worker_conn in zip(worker_gpu_ids, self.worker_conns)
        ]
        for p in self.ps:
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -100000
        self.last_learn_T = 0
