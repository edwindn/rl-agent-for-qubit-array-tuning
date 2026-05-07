## Benchmark run weights

| Algorithm | wandb run | Notes |
|-----------|-----------|-------|
| MAPPO     | `zany-cloud-648` (https://wandb.ai/anon-entity/RLModel/runs/5cz5lz9v) | 4 dots, 400 iters, no virtualization (`update_method: null`). Best ~iter 250-300. |

## Main script for running inference

Example command:
```uv run main.py --load-checkpoint weights/run_xxx --collect-data```

ablation_metrics.py
usage:
* ```--radius```: voltage distance from ground truth required for convergence
  pass a single number or a comma separated list, eg. 0.2,0.5,1,2
* ```--length```: number of steps to consider. note the value will be 49 or 99 since rollouts omit the first datapoint
  use 49 by default, only the virtualisation ablation has 99 timesteps