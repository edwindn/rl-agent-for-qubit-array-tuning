import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ml_collections
import wandb

# Custom imports
from qarray_utils import (
    create_qarray_dataset, 
    QArrayBatchParser, 
    VoltageEncoder, 
    VoltageDecoder,
    train_step,
    eval_step
)


def create_config():
    """Create configuration for quantum array autoencoder training."""
    config = ml_collections.ConfigDict()
    
    # Model configuration
    config.model = ml_collections.ConfigDict()
    config.model.model_name = "qarray_autoencoder"
    
    # Encoder: voltage array -> latent representation
    config.model.encoder = ml_collections.ConfigDict()
    config.model.encoder.input_dim = 2  # TODO ADD OTHER VOLTAGES AND CAPACITANCES
    config.model.encoder.hidden_dims = [256, 128, 64]
    config.model.encoder.latent_dim = 8 # latent space dim
    config.model.encoder.activation = "relu"
    config.model.encoder.dropout_rate = 0.1
    
    # Decoder: latent representation -> voltage array
    config.model.decoder = ml_collections.ConfigDict()
    config.model.decoder.latent_dim = 8
    config.model.decoder.hidden_dims = [64, 128, 256]
    config.model.decoder.output_dim = 2  # Reconstruct voltage array (matching input_dim)
    config.model.decoder.activation = "relu"
    config.model.decoder.dropout_rate = 0.1
    
    # Dataset configuration
    config.dataset = ml_collections.ConfigDict()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config.dataset.data_dir = os.path.join(current_dir, "../data")  # Directory with voltage data
    config.dataset.train_batch_size = 64
    config.dataset.eval_batch_size = 128
    config.dataset.num_workers = 4
    config.dataset.train_split = 0.8
    config.dataset.normalize_voltages = True
    config.dataset.voltage_range = (-10.0, 2.0)  # TODO MUST CHANGE NORMALISATION FOR EACH INDEX
    
    # Training configuration
    config.training = ml_collections.ConfigDict()
    config.training.max_steps = 50000
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-4
    config.training.warmup_steps = 1000
    config.training.beta1 = 0.9
    config.training.beta2 = 0.999
    config.training.reconstruction_weight = 1.0
    config.training.regularization_weight = 0.01  # L2 regularization on latent space
    config.training.clip_grad_norm = 1.0  # Gradient clipping
    
    # Device configuration
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.seed = 42  # Random seed
    
    # Logging configuration
    config.logging = ml_collections.ConfigDict()
    config.logging.log_interval = 100
    config.logging.eval_interval = 1000
    config.logging.plot_interval = 2000
    
    # Saving configuration
    config.saving = ml_collections.ConfigDict()
    config.saving.save_interval = 5000
    config.saving.checkpoint_dir = "./checkpoints"
    
    # W&B configuration
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "qarray_autoencoder"
    config.wandb.entity = None
    
    return config


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Main training and evaluation loop for quantum array autoencoder."""
    
    # Set device and seed
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print(f"Using device: {device}")
    
    # Initialize models
    encoder = VoltageEncoder(
        input_dim=config.model.encoder.input_dim,
        hidden_dims=config.model.encoder.hidden_dims,
        latent_dim=config.model.encoder.latent_dim,
        activation=config.model.encoder.activation,
        dropout_rate=config.model.encoder.dropout_rate
    ).to(device)
    
    decoder = VoltageDecoder(
        latent_dim=config.model.decoder.latent_dim,
        hidden_dims=config.model.decoder.hidden_dims,
        output_dim=config.model.decoder.output_dim,
        activation=config.model.decoder.activation,
        dropout_rate=config.model.decoder.dropout_rate
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"Model storage cost: {total_params * 4 / 1024 / 1024:.2f} MB of parameters")
    
    # Create optimizer with learning rate scheduling
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.max_steps,
        eta_min=config.training.learning_rate * 0.01
    )
    
    # Create datasets and dataloaders
    train_dataset, eval_dataset = create_qarray_dataset(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create batch parser
    batch_parser = QArrayBatchParser(config)
    
    # Setup checkpoint and logging
    job_name = f"{config.model.model_name}_latent_dim_{config.model.encoder.latent_dim}"
    ckpt_path = os.path.join(config.saving.checkpoint_dir, job_name)
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Save config
    config_dict = config.to_dict()
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)
    
    # Initialize W&B
    wandb.init(project=config.wandb.project, name=job_name, config=config)
    
    # Training loop
    best_eval_loss = float('inf')
    step = 0
    
    print("Starting training...")
    
    for epoch in range(config.training.max_steps // len(train_loader) + 1):
        encoder.train()
        decoder.train()
        
        train_losses = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            if step >= config.training.max_steps:
                break
                
            # Process batch and move to device
            voltages = batch_parser.process_batch(batch).to(device)
            
            # Training step
            loss_dict = train_step(
                encoder, decoder, optimizer, voltages, config, 
                clip_grad_norm=config.training.clip_grad_norm
            )
            train_losses.append(loss_dict)
            
            # Update learning rate
            scheduler.step()
            
            step += 1
            
            # Logging
            if step % config.logging.log_interval == 0:
                avg_losses = {}
                for key in train_losses[0].keys():
                    avg_losses[key] = np.mean([loss[key] for loss in train_losses[-10:]])
                
                end_time = time.time()
                log_dict = {
                    "train/reconstruction_loss": avg_losses['reconstruction_loss'],
                    "train/regularization_loss": avg_losses['regularization_loss'],
                    "train/total_loss": avg_losses['total_loss'],
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": step,
                    "train/epoch": epoch
                }
                
                wandb.log(log_dict, step=step)
                # print(f"Step {step}: Train Loss = {avg_losses['total_loss']:.4f}, "
                #       f"Recon = {avg_losses['reconstruction_loss']:.4f}, "
                #       f"Reg = {avg_losses['regularization_loss']:.4f}, "
                #       f"Time = {end_time - start_time:.2f}s")
                start_time = end_time
            
            # Evaluation
            if step % config.logging.eval_interval == 0:
                encoder.eval()
                decoder.eval()
                
                eval_losses = []
                latent_codes = []
                
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        eval_voltages = batch_parser.process_batch(eval_batch).to(device)
                        
                        eval_loss_dict, latent_batch = eval_step(
                            encoder, decoder, eval_voltages, config
                        )
                        eval_losses.append(eval_loss_dict)
                        latent_codes.append(latent_batch.cpu())
                
                # Compute average evaluation metrics
                avg_eval_loss = {}
                for key in eval_losses[0].keys():
                    avg_eval_loss[key] = np.mean([loss[key] for loss in eval_losses])
                
                all_latent_codes = torch.cat(latent_codes, dim=0)
                
                eval_log_dict = {
                    "eval/reconstruction_loss": avg_eval_loss['reconstruction_loss'],
                    "eval/regularization_loss": avg_eval_loss['regularization_loss'],
                    "eval/total_loss": avg_eval_loss['total_loss'],
                    "eval/latent_std": torch.std(all_latent_codes).item(),
                    "eval/latent_mean": torch.mean(torch.abs(all_latent_codes)).item(),
                }
                
                wandb.log(eval_log_dict, step=step)
                # print(f"Eval Loss = {avg_eval_loss['total_loss']:.4f}")
                
                # Save best model
                if avg_eval_loss['total_loss'] < best_eval_loss:
                    best_eval_loss = avg_eval_loss['total_loss']
                    torch.save({
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'step': step,
                        'best_loss': best_eval_loss,
                        'config': config.to_dict()
                    }, os.path.join(ckpt_path, 'best_model.pth'))
                
                encoder.train()
                decoder.train()
            
            # Visualization
            if step % config.logging.plot_interval == 0:
                plot_latent_space(encoder, eval_loader, batch_parser, device, step)
            
            # Save checkpoint
            if step % config.saving.save_interval == 0:
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step,
                    'config': config.to_dict()
                }, os.path.join(ckpt_path, f'checkpoint_step_{step}.pth'))
        
        if step >= config.training.max_steps:
            break
    
    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'config': config.to_dict()
    }, os.path.join(ckpt_path, 'final_model.pth'))


def plot_latent_space(encoder, eval_loader, batch_parser, device, step):
    """Visualize the learned latent space."""
    import matplotlib.pyplot as plt
    
    encoder.eval()
    latent_codes = []
    voltages = []
    
    with torch.no_grad():
        for batch in eval_loader:
            processed_batch = batch_parser.process_batch(batch).to(device)
            latent = encoder(processed_batch)
            
            latent_codes.append(latent.cpu().numpy())
            voltages.append(batch['voltages'].numpy())
    
    latent_codes = np.concatenate(latent_codes, axis=0)
    voltages = np.concatenate(voltages, axis=0)
    
    # Plot 2D latent space (first two dimensions)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.6, s=20)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(f'Latent Space Visualization (Step {step})')
    plt.grid(True, alpha=0.3)
    
    # Plot voltage distribution
    plt.subplot(1, 2, 2)
    plt.hist(voltages.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Voltage Values')
    plt.ylabel('Frequency')
    plt.title('Voltage Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"latent_space": wandb.Image(plt)}, step=step)
    plt.close()


def main():
    """Main entry point."""
    config = create_config()
    train_and_evaluate(config)


if __name__ == "__main__":
    main()