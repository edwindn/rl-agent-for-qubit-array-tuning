import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import os

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)  # Assuming 28x28 input images

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 1, 28, 28)  # Random 28x28 grayscale images
        self.labels = torch.randint(0, 10, (size,))  # Random labels (0-9)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"image": self.data[idx], "label": self.labels[idx]}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Changed to avoid conflicts
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Create dataset and data loader
        dataset = SyntheticDataset(size=1000)
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,  # Reduced to avoid multiprocessing issues
            pin_memory=True
        )

        # Initialize model and wrap with DDP
        model = SimpleCNN().to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Print info on rank 0
        if rank == 0:
            print(f"Dataset size: {len(dataset)}")
            print(f"Using {world_size} GPUs" if torch.cuda.is_available() else "Using CPU")

        # Training loop
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            if rank == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}")
            epoch_loss = 0.0

            progress_bar = tqdm(data_loader, desc="Training", leave=False, disable=rank != 0)
            for batch_idx, data in enumerate(progress_bar):
                images = data["image"].to(device)
                labels = data["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

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
                print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(data_loader):.4f}")

        # Save model on rank 0
        if rank == 0:
            torch.save(model.module.state_dict(), "simple_cnn.pth")
    finally:
        cleanup()

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Test DDP with a simple CNN")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    args = parser.parse_args()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()