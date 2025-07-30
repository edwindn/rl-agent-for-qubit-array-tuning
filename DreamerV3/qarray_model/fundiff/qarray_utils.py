import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any
import glob
import ml_collections


class QArrayDataset(Dataset):
    """Dataset class for quantum array voltage data."""
    
    def __init__(self, data_dir: str, normalize_voltages: bool = True,
                 voltage_range: Tuple[float, float] = (-10.0, 2.0)):
        self.data_dir = data_dir
        self.normalize_voltages = normalize_voltages
        self.voltage_range = voltage_range
        
        # Load all voltage data
        self.voltages = []
        self._load_data()
        
        if self.normalize_voltages:
            self._normalize_voltages()
    
    def _load_data(self):
        """Load voltage data from npz files."""
        data_files = glob.glob(os.path.join(self.data_dir, "*/data.npz"))
        
        for data_file in data_files:
            data = np.load(data_file, allow_pickle=True)
            voltages = data['voltages']
            
            for voltage in voltages:
                self.voltages.append(voltage.astype(np.float32))
        
        self.voltages = np.array(self.voltages)
        print(f"Loaded {len(self.voltages)} voltage samples")
    
    def _normalize_voltages(self):
        """Normalize voltages to specified range."""
        v_min, v_max = self.voltage_range
        # Normalize to [0, 1] then to specified range
        self.voltages = (self.voltages - v_min) / (v_max - v_min)
    
    def __len__(self):
        return len(self.voltages)
    
    def __getitem__(self, idx):
        return {'voltages': torch.tensor(self.voltages[idx], dtype=torch.float32)}


def create_qarray_dataset(config: ml_collections.ConfigDict):
    """Create train and evaluation datasets."""
    dataset = QArrayDataset(
        data_dir=config.dataset.data_dir,
        normalize_voltages=config.dataset.normalize_voltages,
        voltage_range=config.dataset.voltage_range
    )
    
    # Split into train/eval
    train_size = int(len(dataset) * config.dataset.train_split)
    eval_size = len(dataset) - train_size
    
    train_indices = np.random.choice(len(dataset), size=train_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)
    
    train_voltages = dataset.voltages[train_indices]
    eval_voltages = dataset.voltages[eval_indices]
    
    class SubDataset(Dataset):
        def __init__(self, voltages):
            self.voltages = voltages
        
        def __len__(self):
            return len(self.voltages)
        
        def __getitem__(self, idx):
            return {'voltages': torch.tensor(self.voltages[idx], dtype=torch.float32)}
    
    train_dataset = SubDataset(train_voltages)
    eval_dataset = SubDataset(eval_voltages)
    
    return train_dataset, eval_dataset


class QArrayBatchParser:
    """Batch parser for quantum array voltage data."""
    
    def __init__(self, config: ml_collections.ConfigDict):
        self.config = config
        self.voltage_dim = config.model.encoder.input_dim
    
    def process_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Process a batch of voltage data."""
        voltages = batch['voltages']
        
        # Ensure correct shape: (batch_size, voltage_dim)
        if voltages.dim() == 1:
            voltages = voltages.unsqueeze(0)  # Add batch dimension
        
        assert voltages.shape[-1] == self.voltage_dim, \
            f"Expected voltage dim {self.voltage_dim}, got {voltages.shape[-1]}"
        
        return voltages


class VoltageEncoder(nn.Module):
    """Encoder for voltage arrays."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 activation: str = "relu", dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Get activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class VoltageDecoder(nn.Module):
    """Decoder for voltage arrays."""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, 
                 activation: str = "relu", dropout_rate: float = 0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Get activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = latent_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


def train_step(encoder, decoder, optimizer, voltages, config, clip_grad_norm=None):
    """Single training step."""
    optimizer.zero_grad()
    
    # Forward pass
    latent = encoder(voltages)
    reconstructed = decoder(latent)
    
    # Compute losses
    reconstruction_loss = nn.MSELoss()(reconstructed, voltages)
    regularization_loss = torch.mean(latent ** 2) * config.training.regularization_weight
    total_loss = reconstruction_loss + regularization_loss
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 
            clip_grad_norm
        )
    
    optimizer.step()
    
    return {
        'reconstruction_loss': reconstruction_loss.item(),
        'regularization_loss': regularization_loss.item(),
        'total_loss': total_loss.item()
    }


def eval_step(encoder, decoder, voltages, config):
    """Single evaluation step."""
    # Forward pass (no gradients)
    latent = encoder(voltages)
    reconstructed = decoder(latent)
    
    # Compute losses
    reconstruction_loss = nn.MSELoss()(reconstructed, voltages)
    regularization_loss = torch.mean(latent ** 2) * config.training.regularization_weight
    total_loss = reconstruction_loss + regularization_loss
    
    loss_dict = {
        'reconstruction_loss': reconstruction_loss.item(),
        'regularization_loss': regularization_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return loss_dict, latent


def compute_latent_statistics(latent_codes: torch.Tensor) -> Dict[str, float]:
    """Compute statistics of latent codes."""
    return {
        'mean': float(torch.mean(latent_codes)),
        'std': float(torch.std(latent_codes)),
        'min': float(torch.min(latent_codes)),
        'max': float(torch.max(latent_codes)),
        'l2_norm': float(torch.mean(torch.norm(latent_codes, dim=1)))
    }