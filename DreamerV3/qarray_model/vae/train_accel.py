"""
accelerate launch --num_processes=4 train_accel.py
"""

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import time
import traceback
import pickle
import warnings

# Suppress the distributed warnings
warnings.filterwarnings("ignore", message="No device id is provided")
warnings.filterwarnings("ignore", message="using GPU .* as device used by this process")

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from vqvae import create_vqvae2_large, VQVAELoss
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

def train_epoch(model, train_loader, optimizer, criterion, accelerator):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    # Only show progress bar on main process
    progress_bar = tqdm(train_loader, desc="Training", leave=False, 
                       disable=not accelerator.is_main_process)
    
    for batch_idx, data in enumerate(progress_bar):
        images = data['image']
        
        optimizer.zero_grad()
        
        # Forward pass
        with accelerator.autocast():  # For mixed precision
            outputs = model(images)
            loss_dict = criterion(outputs, images)
            loss = loss_dict['total_loss']
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar only on main process
        if accelerator.is_main_process:
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                "Recon Loss": f"{loss_dict['reconstruction_loss'].item():.4f}",
                "VQ Loss": f"{loss_dict['vq_loss'].item():.4f}"
            })
    
    return epoch_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, accelerator):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data in val_loader:
            images = data['image']
            
            with accelerator.autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, images)
                loss = loss_dict['total_loss']
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

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
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Use file-based sharing for dataset
    dataset_cache_path = "/tmp/vae_dataset_cache.pkl"
    
    if accelerator.is_main_process:
        print(f"Starting training with {accelerator.num_processes} processes")
        print(f"Main process loading data...")
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(this_dir, '../data')
        dataset = load_data(data_dir)
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Save dataset to cache file
        with open(dataset_cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset cached for other processes")
    
    # Wait for main process to finish loading and caching
    accelerator.wait_for_everyone()
    
    # Other processes load from cache
    if not accelerator.is_main_process:
        with open(dataset_cache_path, 'rb') as f:
            dataset = pickle.load(f)

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
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(test_dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes}")
    
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

    model = create_vqvae2_large(image_size=128, in_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = VQVAELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Prepare everything with accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        print("Starting training...")

    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, accelerator)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, accelerator)
        
        # Gather validation loss from all processes
        val_loss = accelerator.gather(torch.tensor(val_loss)).mean().item()
        
        # Step scheduler
        scheduler.step()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                accelerator.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, "vqvae_qarray_best.pth")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % args.save_interval == 0:
                accelerator.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f"vqvae_qarray_epoch_{epoch+1}.pth")
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.save(accelerator.unwrap_model(model).state_dict(), "vqvae_qarray_final.pth")
        print("Training completed!")
        
        # Clean up cache file
        if os.path.exists(dataset_cache_path):
            os.remove(dataset_cache_path)

if __name__ == "__main__":
    main()