from collections import defaultdict
import logging
import numpy as np


# Keys for which we emit an EMA-smoothed companion metric to wandb.
# Matches the spirit of the main training's metrics_logger.py "plunger_return_ema" field.
_EMA_KEYS = (
    "return_mean", "test_return_mean",
    "plunger_return_mean", "test_plunger_return_mean",
    "barrier_return_mean", "test_barrier_return_mean",
    "plunger_return_avg_mean", "test_plunger_return_avg_mean",
    "barrier_return_avg_mean", "test_barrier_return_avg_mean",
)


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.use_wandb = False
        self.wandb_run = None
        self.ema_alpha = None
        self._ema_state: dict[str, float] = {}

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_wandb(self, project, entity=None, config=None, name=None, ema_period: int = 20):
        """
        Init a wandb run and route every log_stat call to it.
        EMA smoothing is applied to a curated list of reward keys, matching
        the main training's "plunger_return_ema"-style metrics for direct
        side-by-side comparison in the wandb UI.
        """
        import wandb
        if wandb.run is None:
            wandb.init(project=project, entity=entity, config=config, name=name)
        self.wandb_run = wandb.run
        self.use_wandb = True
        self.ema_alpha = 2.0 / (ema_period + 1.0)

    def _wandb_value(self, value):
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

        if self.use_wandb:
            scalar = self._wandb_value(value)
            if scalar is None:
                return
            payload = {key: scalar}
            if key in _EMA_KEYS:
                prev = self._ema_state.get(key, scalar)
                new_ema = self.ema_alpha * scalar + (1.0 - self.ema_alpha) * prev
                self._ema_state[key] = new_ema
                payload[f"{key}_ema"] = new_ema
            # PPO-compatible alias: vendor appends "_mean" to every stats-bucket key,
            # so our `plunger_return_avg` (set per-step in env_wrapper) lands as
            # `plunger_return_avg_mean`. Also emit it under PPO's name.
            if key.endswith("_return_avg_mean"):
                alias = key.replace("_return_avg_mean", "_return_avg")
                payload[alias] = scalar
            try:
                self.wandb_run.log(payload, step=t)
            except Exception:
                pass

    def log_artifact(self, local_dir: str, name: str, metadata: dict | None = None) -> None:
        """Upload a local directory as a wandb artifact. No-op if wandb isn't set up."""
        if not self.use_wandb or self.wandb_run is None:
            return
        import wandb
        artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
        artifact.add_dir(local_dir)
        self.wandb_run.log_artifact(artifact)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1] if isinstance(x[1], float) else x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
