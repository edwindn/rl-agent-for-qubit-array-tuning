import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from vqvae import create_vqvae2_large, VQVAELoss
from utils import load_data

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Load dataset
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, '../data')
    dataset = load_data(data_dir)

    # Split dataset
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Use DistributedSampler for training data
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=2, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # Print dataset sizes on rank 0
    if rank == 0:
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of test samples: {len(test_loader.dataset)}")
        print(f"Using {world_size} GPUs")

    # Initialize model and wrap with DDP
    VQVAE = create_vqvae2_large(image_size=128, in_channels=1).to(device)
    VQVAE = DDP(VQVAE, device_ids=[rank])
    optimizer = torch.optim.Adam(VQVAE.parameters(), lr=1e-3)
    criterion = VQVAELoss()

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Ensure consistent shuffling across GPUs
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=rank != 0)
        for batch_idx, data in enumerate(progress_bar):
            images = data['image'].to(device).float()
            outputs = VQVAE(images)
            loss_dict = criterion(outputs, images)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if rank == 0:
                progress_bar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}"
                })

        if rank == 0:
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")

    # Save model on rank 0
    if rank == 0:
        torch.save(VQVAE.module.state_dict(), "vqvae_qarray.pth")

    cleanup()

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train VQ-VAE on qarray data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs for training")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()