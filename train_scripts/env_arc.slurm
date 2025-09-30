#!/bin/bash
#SBATCH --job-name=arc_install_conda_packages
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --mail-user=edn.work@proton.me
#SBATCH --mail-type=ALL

mkdir -p logs



log_disk_usage() {
    echo "--------- Disk Usage & Quota ---------"
    du -sh ~
    quota -s
    echo "-------------------------------------"
}

module purge
module load Anaconda3

CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/12\.[0-9\.]+' | sort -V | tail -n 1)
if [[ -z "$CUDA_VERSION" ]]; then
    CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/[0-9\.]+' | sort -V | tail -n 1)
fi

if [[ -z "$CUDA_VERSION" ]]; then
    echo "No CUDA modules found — exiting."
    exit 1
else
    echo "Loading CUDA module: $CUDA_VERSION"
    module load "$CUDA_VERSION"
fi

# log_disk_usage


CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "conda.sh not found — exiting."
    exit 1
fi

# Clean caches before environment creation
conda clean --all -y
pip cache purge

# log_disk_usage

# Create environment
ENV_NAME="rl_train_env"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Activating existing Conda environment '$ENV_NAME'..."
else
    echo "Creating Conda environment '$ENV_NAME' with Python 3.11..."
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"


# torch, ray[rllib], qarray, wandb, torchvision[models]
# scikit-learn, qarray-latched, scienceplots
PACKAGES=(
    "torch"
    "torchvision[models]"
    "ray[rllib]"
    "qarray"
    "wandb[media]"
    "scikit-learn"
    "git+ssh://git@github.com/b-vanstraaten/qarray-latched.git@c076d4cef57a071dd6e52458ad5937589747c18f"
    "scienceplots"
    "jax[cuda12]"
    "python-dotenv"
)

pip install --upgrade pip
for pkg in "${PACKAGES[@]}"; do
    echo "Installing $pkg..."
    pip install "$pkg"
done

# Clean caches after installation
conda clean --all -y
pip cache purge
# log_disk_usage

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not available — GPU may not be allocated correctly."
    exit 1
fi


log_disk_usage