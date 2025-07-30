import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from qarray_utils import VoltageEncoder, VoltageDecoder


def reconstruct(voltages, encoder, decoder):
    with torch.no_grad():
        latent = encoder(voltages)
        reconstructed = decoder(latent)
    return reconstructed.squeeze().cpu().numpy()


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run inference on the qarray model.")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/v1/final_model.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']

    encoder = VoltageEncoder(
        input_dim=config['model']['encoder']['input_dim'],
        hidden_dims=config['model']['encoder']['hidden_dims'],
        latent_dim=config['model']['encoder']['latent_dim'],
        activation=config['model']['encoder']['activation'],
    ).to(device)
    
    decoder = VoltageDecoder(
        latent_dim=config['model']['decoder']['latent_dim'],
        hidden_dims=config['model']['decoder']['hidden_dims'],
        output_dim=config['model']['decoder']['output_dim'],
        activation=config['model']['decoder']['activation'],
    ).to(device)

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    voltage_range = (-10.0, 2.0)
    voltages = np.random.uniform(voltage_range[0], voltage_range[1], (1, config['model']['encoder']['input_dim']))
    voltages = torch.tensor(voltages, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Input voltages: {voltages}")
    reconstructed_voltages = reconstruct(voltages, encoder, decoder)
    print(f"Reconstructed voltages: {reconstructed_voltages}")


if __name__ == "__main__":
    main()