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

python - << 'EOF'
import os
import sys
from pathlib import Path

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# Check if python-dotenv is installed
try:
    import dotenv
    print(f"✅ python-dotenv is installed (version: {dotenv.__version__})")
except ImportError:
    print("❌ python-dotenv is NOT installed")
    sys.exit(1)

print()
print("=" * 60)
print("Current working directory:", os.getcwd())
print("=" * 60)

# Look for .env files in common locations
possible_env_paths = [
    Path.cwd() / ".env",
    Path.cwd().parent / ".env",
    Path.cwd().parent.parent / ".env",
    Path.home() / ".env",
]

print("\nSearching for .env files:")
for path in possible_env_paths:
    if path.exists():
        print(f"  ✅ Found: {path}")
    else:
        print(f"  ❌ Not found: {path}")

print()
print("=" * 60)
print("Testing dotenv.load_dotenv():")
print("=" * 60)

# Test loading from current directory
from dotenv import load_dotenv, find_dotenv

found_path = find_dotenv(usecwd=True)
if found_path:
    print(f"✅ find_dotenv() found: {found_path}")
else:
    print("❌ find_dotenv() did not find any .env file")

print()
result = load_dotenv(verbose=True)
print(f"\nload_dotenv() returned: {result}")

print()
print("=" * 60)
print("Environment variables after load_dotenv():")
print("=" * 60)

# Check for specific environment variables
test_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "PATH", "HOME"]
for var in test_vars:
    value = os.getenv(var)
    if value:
        # Mask sensitive values
        if "KEY" in var or "TOKEN" in var or "SECRET" in var:
            masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"  {var}: {masked_value}")
        else:
            print(f"  {var}: {value}")
    else:
        print(f"  {var}: <not set>")

print()
print("=" * 60)
print("All environment variables starting with WANDB_:")
print("=" * 60)
wandb_vars = {k: v for k, v in os.environ.items() if k.startswith("WANDB_")}
if wandb_vars:
    for key, value in sorted(wandb_vars.items()):
        if "KEY" in key or "TOKEN" in key or "SECRET" in key:
            masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"  {key}: {masked_value}")
        else:
            print(f"  {key}: {value}")
else:
    print("  <no WANDB_* variables found>")

print()
print("✅ Dotenv loading test completed successfully!")
EOF

echo "============================="