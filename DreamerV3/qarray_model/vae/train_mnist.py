import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bar
import wandb

from vqvae_mnist import create_vqvae2_large, ConditionalVQVAELoss

def load_mnist(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]t
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
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs for training")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU device to train on (for single-GPU)")
    parser.add_argument("--use_wandb", action='store_true', help="Use wandb for logging")
    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    use_wandb = args.use_wandb
    train_loader, test_loader = load_mnist(batch_size=batch_size)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    VQVAE = create_vqvae2_large(image_size=28, in_channels=1).to(device)
    optimizer = torch.optim.Adam(VQVAE.parameters(), lr=1e-3)
    criterion = ConditionalVQVAELoss().to(device)

    if use_wandb:
        wandb.init(project="vqvae-mnist")

    # TODO: increase classification loss weight over time

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        # rewrite since we just keep the dataset from the latest epoch
        label_dataset = []
        top_latents_dataset = []
        bottom_latents_dataset = []

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            labels = F.one_hot(labels, num_classes=10)
            label_dataset.append(labels)

            images = images.to(device)
            labels = labels.to(device)
            outputs = VQVAE(images, labels)
            loss_dict = criterion(outputs, images, labels)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}"
            })

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "total_loss": loss.item(),
                    "reconstruction_loss": loss_dict['reconstruction_loss'].item(),
                    "vq_loss": loss_dict['vq_loss'].item(),
                    "classification_loss": loss_dict['classification_loss'].item()
                })

            top_latents_dataset.append(outputs['top_indices'])
            bottom_latents_dataset.append(outputs['bottom_indices'])

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")
        
        labels_dataset = torch.cat(label_dataset, dim=0)
        top_latents_dataset = torch.cat(top_latents_dataset, dim=0)
        bottom_latents_dataset = torch.cat(bottom_latents_dataset, dim=0)
        torch.save({
            'labels': labels_dataset,
            'top_latents': top_latents_dataset,
            'bottom_latents': bottom_latents_dataset
        }, f"vqvae_mnist_dataset.pth")


    torch.save(VQVAE.state_dict(), "vqvae_mnist.pth")

if __name__ == "__main__":
    main()