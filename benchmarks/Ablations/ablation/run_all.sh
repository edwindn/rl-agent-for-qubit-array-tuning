#!/usr/bin/env bash
# Launch ablation evals in parallel, one per available GPU.
#
# Usage:
#   ./run_all.sh                # run every algo with pipeline=rlmodel
#   ./run_all.sh algo1 algo2    # run only those algos
#
# Each algo runs as a detached nohup process; logs go to /tmp/ablation_<algo>.log.
# After all complete, run compute_table.py to aggregate.
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# Excludes GPU 0 (typically busy on this server).
GPUS=(1 2 3 4 5 6 7)

if [ "$#" -eq 0 ]; then
  ALGOS=$(uv run python -c "
import yaml
cfg = yaml.safe_load(open('ablation_config.yaml'))
print(' '.join(k for k, v in cfg['algos'].items() if v.get('pipeline','rlmodel')=='rlmodel'))
")
else
  ALGOS="$*"
fi

i=0
for algo in $ALGOS; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  log="/tmp/ablation_${algo}.log"
  echo "Launching $algo on GPU $gpu -> $log"
  nohup bash -c "cd <repo> && uv run python $SCRIPT_DIR/run_ablation.py --algo $algo --gpu $gpu" \
    > "$log" 2>&1 < /dev/null &
  disown
  echo "  PID=$!"
  i=$((i + 1))
done
echo
echo "All launched. Tail with:  tail -f /tmp/ablation_*.log"
echo "Aggregate with:           cd $SCRIPT_DIR && uv run python compute_table.py"
