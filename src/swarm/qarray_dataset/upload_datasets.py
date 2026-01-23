from huggingface_hub import HfApi, login
import os

# Login to HuggingFace
login()

# Dataset paths (relative to this script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets = {
    "barrier_dataset_1": os.path.join(script_dir, "barrier_dataset_1"),
    "barrier_dataset_2": os.path.join(script_dir, "barrier_dataset_2"),
    "no_barrier_dataset_1": os.path.join(script_dir, "no_barrier_dataset_1"),
    "no_barrier_dataset_2": os.path.join(script_dir, "no_barrier_dataset_2"),
}

api = HfApi()

for dataset_name, dataset_path in datasets.items():
    repo_id = f"edwindn/{dataset_name}"

    print(f"\nUploading {dataset_name}...")

    # Create repository
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    # Upload the entire dataset folder
    api.upload_folder(
        folder_path=dataset_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"✓ Dataset uploaded successfully to {repo_id}")

print("\nAll datasets uploaded successfully!")
