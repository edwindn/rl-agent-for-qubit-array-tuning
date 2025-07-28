import numpy as np
import torch
import torch.nn as nn
import os
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.models import AutoencoderKL
from dotenv import load_dotenv

load_dotenv()

def load_model():
    """
    Loads the Stable Diffusion 3 model components for fine-tuning.
    Returns only the trainable components (VAE and Transformer) without tokenizers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # Load the full pipeline first to get the components
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype,
            cache_dir="./model_cache"
        )
        
        # Extract individual components for training
        # VAE (Variational Autoencoder) - for encoding/decoding images
        vae = pipeline.vae
        vae.requires_grad_(True)  # Enable gradients for training
        
        # Transformer (Main diffusion model) - core model to fine-tune
        transformer = pipeline.transformer
        transformer.requires_grad_(True)  # Enable gradients for training
        
        # Scheduler (for noise scheduling during training)
        scheduler = pipeline.scheduler
        
        # Move models to device
        vae = vae.to(device)
        transformer = transformer.to(device)
        
        print(f"Models loaded successfully on {device}")
        print(f"VAE parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad):,}")
        print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")
        
        return {
            'vae': vae,
            'transformer': transformer,
            'scheduler': scheduler,
            'device': device,
            'dtype': dtype
        }
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        raise Exception("Could not load Stable Diffusion 3 model for training")


def load_data(data_dir):
    """
    Load training data from the specified directory.
    """
    dataset = []

    shard_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for f in shard_folders:
        data_path = os.path.join(data_dir, f, 'data.npz')
        data = np.load(data_path, allow_pickle=True)
        voltages, images = data['voltages'], data['states']
        for v, i in zip(voltages, images):
            dataset.append({'voltages': v, 'image': i})

    print(f"Loaded {len(dataset)} samples from {data_dir}")
    return dataset


def setup_training(model_components, learning_rate=1e-5):
    """
    Setup optimizers and loss functions for training.
    """
    vae = model_components['vae']
    transformer = model_components['transformer']
    
    # Set up optimizers
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
    transformer_optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
    
    # Loss function (MSE for diffusion training)
    criterion = nn.MSELoss()
    
    return {
        'vae_optimizer': vae_optimizer,
        'transformer_optimizer': transformer_optimizer,
        'criterion': criterion
    }


def train_step(model_components, training_components, batch_data):
    """
    Perform one training step.
    """
    vae = model_components['vae']
    transformer = model_components['transformer']
    scheduler = model_components['scheduler']
    device = model_components['device']
    
    vae_optimizer = training_components['vae_optimizer']
    transformer_optimizer = training_components['transformer_optimizer']
    criterion = training_components['criterion']
    
    # Training step implementation would go here
    # This is a placeholder for the actual training logic
    
    print("Training step not implemented yet")
    return 0.0  # Return loss


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train model on CSD data")
    parser.add_argument("--data_dir", type=str, default='./data', help="Directory containing training data")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_components = load_model()
    
    training_components = setup_training(model_components, args.learning_rate)
    
    data = load_data(args.data_dir)
    
    print("Training setup complete!")
    exit()

    # Training loop would go here
    # for epoch in range(args.num_epochs):
    #     for batch in data_loader:
    #         loss = train_step(model_components, training_components, batch)
    #         print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == '__main__':
    main()