import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from vae.vqvae import create_vqvae2_large, VQVAELoss
from utils import load_data

"""
Run from parent directory with DDP:
torchrun --nproc_per_node=8 -m vae.train
"""

def setup(rank, world_size):
    """Initialize the process group for DDP."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use localhost or the machine's IP address
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the process group."""
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, criterion, device, rank):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
    else:
        progress_bar = train_loader
    
    for batch_idx, data in enumerate(progress_bar):
        images = data['image'].to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss_dict = criterion(outputs, images)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar only on rank 0
        if rank == 0:
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                "Recon Loss": f"{loss_dict['reconstruction_loss'].item():.4f}",
                "VQ Loss": f"{loss_dict['vq_loss'].item():.4f}"
            })
    
    return epoch_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data in val_loader:
            images = data['image'].to(device).float()
            outputs = model(images)
            loss_dict = criterion(outputs, images)
            loss = loss_dict['total_loss']
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def main_worker(rank, world_size, args, train_dataset, test_dataset):
    """Main worker function for each GPU process."""
    print(f"Running DDP on rank {rank}/{world_size}")
    
    # Setup DDP
    setup(rank, world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Important for DDP
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(test_dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
    
    # Create model
    model = create_vqvae2_large(image_size=128, in_channels=1).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = VQVAELoss()
    
    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, rank)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Synchronize validation loss across all processes
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size
        
        # Step scheduler
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, "vqvae_qarray_best.pth")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f"vqvae_qarray_epoch_{epoch+1}.pth")
    
    # Save final model
    if rank == 0:
        torch.save(model.module.state_dict(), "vqvae_qarray_final.pth")
        print("Training completed!")
    
    cleanup()

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Train VQ-VAE on qarray data with DDP")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Batch size per GPU for training")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--world_size", type=int, default=2,
                       help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Load data once on the CPU
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, '../data')
    dataset = load_data(data_dir)
    
    # Split dataset into training and validation sets
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Ensure same split across all processes
    )
    
    # Launch DDP training
    mp.spawn(main_worker, args=(args.world_size, args, train_dataset, test_dataset), 
             nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()