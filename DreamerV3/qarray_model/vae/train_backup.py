import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import time
import traceback
import pickle

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae.vqvae import create_vqvae2_large, VQVAELoss
from utils import load_data

class TensorDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to ensure your dataset returns proper tensors"""
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convert to proper tensor format
        if isinstance(item, dict):
            # Ensure all values are tensors
            result = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value
                else:
                    result[key] = torch.tensor(value)
            return result
        else:
            # If it's not a dict, convert to dict format
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return {
                    'image': torch.tensor(item[1]) if not isinstance(item[1], torch.Tensor) else item[1],
                    'voltages': torch.tensor(item[0]) if not isinstance(item[0], torch.Tensor) else item[0]
                }
            else:
                raise ValueError(f"Unexpected item format: {type(item)}")

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Train VQ-VAE on qarray data with Accelerate")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Batch size per GPU for training")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    args = parser.parse_args()
    
    # Initialize accelerator first
    accelerator = Accelerator()
    
    # Use file-based sharing for dataset
    dataset_cache_path = "/tmp/vae_dataset_cache.pkl"
    
    if accelerator.is_main_process:
        print(f"Main process {accelerator.process_index} loading data...")
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(this_dir, '../data')
        dataset = load_data(data_dir)
        print(f"Main process loaded dataset of type {type(dataset)} and length {len(dataset)}")
        
        # Save dataset to cache file
        with open(dataset_cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset cached to file.")
    
    # Wait for main process to finish loading and caching
    accelerator.wait_for_everyone()
    
    # Other processes load from cache
    if not accelerator.is_main_process:
        with open(dataset_cache_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Process {accelerator.process_index} loaded dataset from cache, length {len(dataset)}")

    # Wrap your dataset to ensure proper tensor format
    wrapped_dataset = TensorDatasetWrapper(dataset)
    
    # Split dataset
    train_size = int(0.99 * len(wrapped_dataset))
    test_size = len(wrapped_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        wrapped_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    if accelerator.is_main_process:
        print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

    # Check what the dataset returns (only on main process to reduce output)
    if accelerator.is_main_process:
        print("DEBUG: Checking what train_dataset[0] returns...")
        sample = train_dataset[0]
        print(f"Dataset sample type: {type(sample)}")
        if isinstance(sample, dict):
            print(f"Dataset sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Check batch format (only on main process)
    if accelerator.is_main_process:
        print("DEBUG: Iterating through first batch of train_loader...")
        for i, batch in enumerate(train_loader):
            print(f"DEBUG: Batch {i} type: {type(batch)}")
            if isinstance(batch, dict):
                print(f"DEBUG: Batch {i} keys: {list(batch.keys())}")
                for key, value in batch.items():
                    print(f"  {key}: {type(value)}, shape: {value.shape}")
            break
        print("DEBUG: Train loader iteration successful.")

    model = create_vqvae2_large(image_size=128, in_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = VQVAELoss()

    if accelerator.is_main_process:
        print("DEBUG: Calling accelerator.prepare...")
    t0 = time.time()
    try:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        if accelerator.is_main_process:
            print(f"DEBUG: accelerator.prepare completed in {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"ERROR: Exception during accelerator.prepare on process {accelerator.process_index}!")
        traceback.print_exc()
        exit(1)

    if accelerator.is_main_process:
        print("SUCCESS: accelerator.prepare worked with real dataset!")
    
    # Add your training loop here
    for epoch in range(1):  # Just test one epoch
        if accelerator.is_main_process:
            print(f"Testing epoch {epoch + 1}")
        
        for i, batch in enumerate(train_loader):
            images = batch['image']
            if accelerator.is_main_process:
                print(f"Batch {i}: image shape {images.shape}")
            if i >= 2:  # Just test a few batches
                break
        break

if __name__ == "__main__":
    main()