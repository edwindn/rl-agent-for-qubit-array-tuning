#!/bin/bash
#SBATCH --job-name=arc_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --mail-user=edn.work@proton.me
#SBATCH --mail-type=ALL

mkdir -p logs

echo "========= SLURM ENVIRONMENT ========="
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo "====================================="

# -----------------------------
# Load Anaconda & CUDA
# -----------------------------
module purge
module load Anaconda3

CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/12\.[0-9\.]+' | sort -V | tail -n 1)
if [[ -z "$CUDA_VERSION" ]]; then
    CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/[0-9\.]+' | sort -V | tail -n 1)
fi

if [[ -z "$CUDA_VERSION" ]]; then
    echo "❌ No CUDA modules found — exiting."
    exit 1
else
    echo "✅ Loading CUDA module: $CUDA_VERSION"
    module load "$CUDA_VERSION"
fi

# -----------------------------
# Enable Conda & Activate Env
# -----------------------------
CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "❌ conda.sh not found — exiting."
    exit 1
fi

ENV_NAME="rl_train_env"
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME" || { echo "❌ Failed to activate environment"; exit 1; }

echo "✅ Using Python: $(which python)"
echo "✅ Python version: $(python --version)"

# -----------------------------
# GPU Check
# -----------------------------
echo "========= GPU CHECK ========="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "❌ nvidia-smi not available — GPU may not be allocated correctly."
    exit 1
fi

python - << 'EOF'
try:
    import torch
    print("Torch CUDA Available:", torch.cuda.is_available())
    print("Torch Device Count:", torch.cuda.device_count())
    print("Torch Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
except ModuleNotFoundError:
    print("❌ Torch not installed in this environment")
EOF
echo "============================="

# -----------------------------
# Run Training Script
# -----------------------------
echo "🚀 Launching training..."
# srun python src/swarm/training/train.py
# srun may interfere with Ray's DDP, trying without
python src/swarm/training/train.py
