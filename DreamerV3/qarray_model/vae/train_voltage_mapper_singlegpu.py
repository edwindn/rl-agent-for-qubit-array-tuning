import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset
from typing import Dict
import os
from tqdm import tqdm
import wandb

from vqvae import create_vqvae2_large
from voltage_mapper import VoltagePixelCNN, VoltageMapperLoss


def train(train: TensorDataset, test: TensorDataset, config: Dict, device: torch.device) -> None:
    vqvae = create_vqvae2_large(**config['vqvae_params']).to(device)
    vqvae.eval()

    model = VoltagePixelCNN(**config['pixel_cnn_params']).to(device)

    use_wandb = config.get('use_wandb')

    criterion = VoltageMapperLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=config['batch_size'], shuffle=False)

    if use_wandb:
        wandb.init(project="voltage-mapper", config=config)

    for epoch in range(config['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", unit="batch")

        for (images, voltages) in pbar:
            images = images.to(device).float()
            voltages = voltages.to(device).float()

            with torch.no_grad():
                outputs = vqvae(images)
            tq, bq = outputs['top_indices'], outputs['bottom_indices']
            targets = {
                'top_indices': tq,
                'bottom_indices': bq
            }

            preds = model(voltages)
            loss_dict = criterion(preds, targets)

            loss = loss_dict['total_loss']
            top_loss = loss_dict['top_loss']
            bottom_loss = loss_dict['bottom_loss']
            top_accuracy = loss_dict['top_accuracy']
            bottom_accuracy = loss_dict['bottom_accuracy']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/total_loss': loss.item(),
                    'train/top_loss': top_loss.item(),
                    'train/bottom_loss': bottom_loss.item(),
                })

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }, f"{config['save_dir']}/checkpoint_epoch_{epoch + 1}.pth")

        model.eval()

        for (images, voltages) in test_loader:
            images = images.to(device).float()
            voltages = voltages.to(device).float()

            with torch.no_grad():
                outputs = vqvae(images)
            tq, bq = outputs['top_indices'], outputs['bottom_indices']
            targets = {
                'top_indices': tq,
                'bottom_indices': bq
            }

            preds = model(voltages)
            loss_dict = criterion(preds, targets)

            if use_wandb:
                wandb.log({
                    'eval/total_loss': loss_dict['total_loss'].item(),
                    'eval/top_loss': loss_dict['top_loss'].item(),
                    'eval/bottom_loss': loss_dict['bottom_loss'].item(),
                })

        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss.item()}")


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Learn voltage to latent mapping")
    parser.add_argument("--voltage_dim", type=int, default=2, help="Number of voltage inputs")
    parser.add_argument("--top_size", type=int, default=16, help="Size of top matrix")
    parser.add_argument("--bottom_size", type=int, default=32, help="Size of bottom matrix")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to use for training")
    parser.add_argument("--wandb", action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument("--save_dir", type=str, default="./voltage_mapper_checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--device_idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("--data_dir", type=str, default="checkpoints3/qarray_dataset_5.pth", help="Directory containing the dataset")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    config = {
        'vqvae_params': {
            'image_size': 128,
            'in_channels': 1,
        },
        'pixel_cnn_params': {
            'voltage_dim': args.voltage_dim,
            'top_matrix_size': args.top_size,
            'bottom_matrix_size': args.bottom_size,
            'embedding_dim': args.embedding_dim
        },
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'use_wandb': args.wandb,
        'save_dir': args.save_dir,
    }

    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    train_dataset, test_dataset = load_dataset(data_dir, num_samples=args.num_samples) # should not be shuffled so far

    print('Training ...')
    train(train_dataset, test_dataset, config, device)


if __name__ == "__main__":
    main()