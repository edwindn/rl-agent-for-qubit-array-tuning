import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

from vqvae import create_vqvae2_large, VQVAELoss

def load_mnist(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers  # Enable shuffling here
    )

    return train_loader, test_loader

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train VQ-VAE on MNIST")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs for training")
    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    train_loader, test_loader = load_mnist(batch_size=batch_size)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    VQVAE = create_vqvae2_large(image_size=28, in_channels=1).to(device)
    optimizer = torch.optim.Adam(VQVAE.parameters(), lr=1e-3)
    criterion = VQVAELoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            outputs = VQVAE(images)
            loss_dict = criterion(outputs, images)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}"
            })

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")


    torch.save(VQVAE.state_dict(), "vqvae_mnist.pth")

if __name__ == "__main__":
    main()