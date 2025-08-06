import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bar
import wandb

from vqvae_mnist import create_vqvae2_large, ConditionalVQVAELoss
from vae import create_vae_large, VAELoss

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
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, test_loader

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train VQ-VAE on MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to train on (for single-GPU)")
    parser.add_argument("--wandb", action='store_true', help="Use wandb for logging")
    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    use_wandb = args.wandb
    train_loader, test_loader = load_mnist(batch_size=batch_size)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model = create_vqvae2_large(image_size=28, in_channels=1).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = ConditionalVQVAELoss().to(device)
    model = create_vae_large(image_size=28, in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = VAELoss().to(device)

    if use_wandb:
        wandb.init(project="vae-mnist")

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        if epoch == num_epochs - 1:
            label_dataset = []
            latents_dataset = []

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            if epoch == num_epochs - 1:
                label_dataset.append(labels)

            images = images.to(device)
            outputs = model(images)
            loss_dict = criterion(outputs, images, labels.float())
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
                    "train/total_loss": loss.item(),
                    "train/reconstruction_loss": loss_dict['reconstruction_loss'].item(),
                    "train/kl_loss": loss_dict['kl_loss'].item(),
                    "train/consistency_loss": loss_dict['consistency_loss'].item()
                })

            if epoch == num_epochs - 1:
                
                latents_dataset.append(outputs['z'].detach().cpu())

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")
        
        if epoch == num_epochs - 1:
            label_dataset = torch.cat(label_dataset, dim=0)
            latents_dataset = torch.cat(latents_dataset, dim=0)

            torch.save({
                'labels': label_dataset,
                'latents': latents_dataset,
            }, f"vae_mnist_dataset.pth")

        
        if use_wandb:
            for (images, labels) in test_loader:
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images)
                    loss_dict = criterion(outputs, images, labels.float())

                wandb.log({
                    "eval/total_loss": loss_dict['total_loss'].item(),
                    "eval/reconstruction_loss": loss_dict['reconstruction_loss'].item(),
                    "eval/kl_loss": loss_dict['kl_loss'].item(),
                    "eval/consistency_loss": loss_dict['consistency_loss'].item()
                })


    torch.save(model.state_dict(), "vae_mnist.pth")

if __name__ == "__main__":
    main()