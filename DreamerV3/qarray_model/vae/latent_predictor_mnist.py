import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class PixelCNN(nn.Module):
    def __init__(self, num_classes, small_dim, large_dim, hidden_dim=128, num_layers=5, num_head_layers=5):
        super(PixelCNN, self).__init__()
        self.num_classes = num_classes
        self.small_dim = small_dim
        self.large_dim = large_dim

        # Project 10D label vector to hidden_dim
        self.label_proj = nn.Linear(num_classes, hidden_dim)

        # Shared trunk
        layers = []
        in_dim = hidden_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        top_layers = []
        for _ in range(num_head_layers-1):
            top_layers.append(nn.Linear(hidden_dim, hidden_dim))
            top_layers.append(nn.ReLU())
            top_layers.append(nn.Dropout(0.1))
        top_layers.append(nn.Linear(hidden_dim, small_dim * small_dim))
        self.top_head = nn.Sequential(*top_layers)

        bottom_layers = []
        for _ in range(num_head_layers-1):
            bottom_layers.append(nn.Linear(hidden_dim, hidden_dim))
            bottom_layers.append(nn.ReLU())
            bottom_layers.append(nn.Dropout(0.1))
        bottom_layers.append(nn.Linear(hidden_dim, large_dim * large_dim))
        self.bottom_head = nn.Sequential(*bottom_layers)

    def forward(self, labels):
        x = F.relu(self.label_proj(labels))
        x = self.trunk(x)
        top_latent = self.top_head(x).view(-1, 1, self.small_dim, self.small_dim)
        bottom_latent = self.bottom_head(x).view(-1, 1, self.large_dim, self.large_dim)
        return top_latent, bottom_latent


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU device to train on (for single-GPU)")
    parser.add_argument("--top_latent_weight", type=float, default=1.0, help="Weight for top latent loss")
    parser.add_argument("--bottom_latent_weight", type=float, default=1.0, help="Weight for bottom latent loss")
    parser.add_argument("--use_wandb", action='store_true', help="Use wandb for logging")
    parser.add_argument("--infer", action='store_true', help="Run inference instead of training")
    args = parser.parse_args()

    if args.infer:
        infer()
        quit()

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load("vqvae_mnist_dataset.pth")
    labels = data['labels']
    top_latents = data['top_latents'].float()
    bottom_latents = data['bottom_latents'].float()

    max_latent_value = 1024 - 1

    top_latents = top_latents / max_latent_value
    bottom_latents = bottom_latents / max_latent_value

    dataset = torch.utils.data.TensorDataset(labels, top_latents, bottom_latents)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cnn = PixelCNN(num_classes=10, small_dim=top_latents.shape[2], large_dim=bottom_latents.shape[2]).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=float(args.lr))
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        cnn.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for i, batch in enumerate(pbar):
            batch_labels, batch_top_latents, batch_bottom_latents = batch
            batch_labels = batch_labels.to(device)
            batch_top_latents = batch_top_latents.to(device)
            batch_bottom_latents = batch_bottom_latents.to(device)

            optimizer.zero_grad()
            top_pred, bottom_pred = cnn(batch_labels.float())
            top_loss = criterion(top_pred.squeeze(), batch_top_latents)
            bottom_loss = criterion(bottom_pred.squeeze().squeeze(), batch_bottom_latents)
            loss = top_loss * args.top_latent_weight + bottom_loss * args.bottom_latent_weight
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({
                "top_loss": f"{top_loss.item():.4f}",
                "bottom_loss": f"{bottom_loss.item():.4f}",
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / (i + 1):.4f}"
            })

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(labels):.8f}")

    torch.save(cnn.state_dict(), "pixel_cnn_mnist.pth")


def infer(path="pixel_cnn_mnist.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelCNN(num_classes=10, small_dim=7, large_dim=14).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    max_latent_value = 1024 - 1
    idx = 4
    
    data = torch.load("vqvae_mnist_dataset.pth")
    labels = data['labels']
    print(labels[:idx])
    labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)
    tops = data['top_latents'].float().to(device)[:idx]
    bottoms = data['bottom_latents'].float().to(device)[:idx]
    inputs = labels_one_hot[:idx]
    inputs = inputs.to(device)
    with torch.no_grad():
        top_latents, bottom_latents = model(inputs)

    print(top_latents * max_latent_value)
    print(tops)



if __name__ == "__main__":
    main()