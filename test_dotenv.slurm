#!/bin/bash
#SBATCH --job-name=test_dotenv_loading
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --mail-user=edn.work@proton.me
#SBATCH --mail-type=ALL

mkdir -p logs

echo "========= SLURM ENVIRONMENT ========="
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo "====================================="

# -----------------------------
# 1️⃣ Load Anaconda
# -----------------------------
module purge
module load Anaconda3

# -----------------------------
# 2️⃣ Enable Conda & Activate Env
# -----------------------------
CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "❌ conda.sh not found — exiting."
    exit 1
fi

ENV_NAME="rl_train_env"
if conda env list | grep -q "$ENV_NAME"; then
    echo "✅ Activating existing Conda environment '$ENV_NAME'..."
    conda activate "$ENV_NAME"
else
    echo "❌ Environment '$ENV_NAME' not found — exiting."
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# -----------------------------
# 3️⃣ Test dotenv loading
# -----------------------------
echo "========= DOTENV LOADING TEST ========="

python -c "
import os
from dotenv import load_dotenv

print('Working directory:', os.getcwd())

load_dotenv()

api_key = os.getenv('WANDB_API_KEY')
if api_key:
    print(f'✅ WANDB_API_KEY found ({len(api_key)} chars)')
else:
    print('❌ WANDB_API_KEY not found')
"

echo "============================="