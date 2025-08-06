import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from torch.utils.data import TensorDataset


class VoltagePixelCNN(nn.Module):
    """
    Uses a combination of MLPs, CNNs, and attention mechanisms to map voltages
    to spatial embedding patterns.
    """
    
    def __init__(
        self,
        voltage_dim: int,
        top_matrix_size: int,
        bottom_matrix_size: int,
        embedding_dim: int = 1024,
        hidden_dims: list = [256, 512, 1024, 2048],
        num_residual_blocks: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_attention: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.voltage_dim = voltage_dim
        self.top_matrix_size = top_matrix_size
        self.bottom_matrix_size = bottom_matrix_size
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Initial voltage embedding
        self.voltage_embedding = nn.Sequential(
            nn.Linear(voltage_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Multi-layer feature extractor
        self.feature_extractor = self._build_feature_extractor(
            hidden_dims, dropout, use_batch_norm
        )
        
        # Residual blocks for complex pattern learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], dropout, use_batch_norm, self.activation)
            for _ in range(num_residual_blocks)
        ])
        
        # Self-attention for global context (optional)
        if use_attention:
            self.self_attention = MultiHeadAttention(
                hidden_dims[-1], num_attention_heads, dropout
            )
        
        # Separate prediction heads for top and bottom matrices
        self.top_head = MatrixPredictionHead(
            hidden_dims[-1], top_matrix_size, embedding_dim, dropout, use_batch_norm, self.activation
        )
        
        self.bottom_head = MatrixPredictionHead(
            hidden_dims[-1], bottom_matrix_size, embedding_dim, dropout, use_batch_norm, self.activation
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_feature_extractor(self, hidden_dims: list, dropout: float, use_batch_norm: bool) -> nn.Module:
        layers = []
        
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]) if use_batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout)
            ])
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, voltages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            voltages: Input voltages [batch_size, voltage_dim]
            
        Returns:
            Dictionary containing:
                - top_indices: [batch_size, top_matrix_size, top_matrix_size]
                - bottom_indices: [batch_size, bottom_matrix_size, bottom_matrix_size]
                - top_logits: [batch_size, top_matrix_size, top_matrix_size, embedding_dim]
                - bottom_logits: [batch_size, bottom_matrix_size, bottom_matrix_size, embedding_dim]
        """
        batch_size = voltages.shape[0]
        
        x = self.voltage_embedding(voltages)
        x = self.feature_extractor(x)
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        if self.use_attention:
            # Reshape for attention: [batch_size, 1, hidden_dims[-1]]
            x_attn = x.unsqueeze(1)
            x_attn = self.self_attention(x_attn, x_attn, x_attn)
            x = x_attn.squeeze(1) + x  # Residual connection
        
        top_output = self.top_head(x)
        bottom_output = self.bottom_head(x)
        
        return {
            'top_indices': top_output['indices'],
            'bottom_indices': bottom_output['indices'],
            'top_logits': top_output['logits'],
            'bottom_logits': bottom_output['logits']
        }


class ResidualBlock(nn.Module):    
    def __init__(self, dim: int, dropout: float, use_batch_norm: bool, activation: nn.Module):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity(),
        )
        
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiHeadAttention(nn.Module):    
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, N, C = query.shape
        
        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MatrixPredictionHead(nn.Module):    
    def __init__(
        self,
        input_dim: int,
        matrix_size: int,
        embedding_dim: int,
        dropout: float,
        use_batch_norm: bool,
        activation: nn.Module
    ):
        super().__init__()
        
        self.matrix_size = matrix_size
        self.embedding_dim = embedding_dim
        
        # Feature expansion for spatial generation
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2) if use_batch_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout)
        )
        
        # Spatial feature generator
        self.spatial_generator = nn.Sequential(
            nn.Linear(input_dim * 2, matrix_size * matrix_size * 64),
            nn.BatchNorm1d(matrix_size * matrix_size * 64) if use_batch_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout)
        )
        
        # Convolutional refinement layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            activation,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            activation,
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            activation,
        )
        
        # Final prediction layer
        self.prediction_layer = nn.Conv2d(128, embedding_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate square matrix of embedding indices.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - indices: [batch_size, matrix_size, matrix_size]
                - logits: [batch_size, matrix_size, matrix_size, embedding_dim]
        """
        batch_size = x.shape[0]
        
        # Feature expansion
        x = self.feature_expansion(x)
        
        # Generate spatial features
        spatial_features = self.spatial_generator(x)
        spatial_features = spatial_features.view(
            batch_size, 64, self.matrix_size, self.matrix_size
        )
        
        # Convolutional refinement
        refined_features = self.conv_layers(spatial_features)
        
        # Final prediction
        logits = self.prediction_layer(refined_features)  # [B, embedding_dim, H, W]
        logits = logits.permute(0, 2, 3, 1)  # [B, H, W, embedding_dim]
        
        # Get indices (argmax over embedding dimension)
        indices = torch.argmax(logits, dim=-1)  # [B, H, W]
        
        return {
            'indices': indices,
            'logits': logits
        }


class VoltageMapperLoss(nn.Module):    
    def __init__(self, weight_top: float = 1.0, weight_bottom: float = 1.0):
        # could experiment with higher top weight since this is the layer with consistency being imposed

        super().__init__()
        self.weight_top = weight_top
        self.weight_bottom = weight_bottom
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth indices
                - top_indices: [batch_size, top_size, top_size]
                - bottom_indices: [batch_size, bottom_size, bottom_size]
        """
        # Top matrix loss
        top_logits = predictions['top_logits']  # [B, H, W, embedding_dim]
        top_targets = targets['top_indices']    # [B, H, W]
        
        # Reshape for cross entropy: [B*H*W, embedding_dim] and [B*H*W]
        top_logits_flat = top_logits.reshape(-1, top_logits.shape[-1])
        top_targets_flat = top_targets.reshape(-1)

        top_loss = self.ce_loss(top_logits_flat, top_targets_flat)
        
        # Bottom matrix loss
        bottom_logits = predictions['bottom_logits']  # [B, H, W, embedding_dim]
        bottom_targets = targets['bottom_indices']    # [B, H, W]
        
        bottom_logits_flat = bottom_logits.reshape(-1, bottom_logits.shape[-1])
        bottom_targets_flat = bottom_targets.reshape(-1)
        
        bottom_loss = self.ce_loss(bottom_logits_flat, bottom_targets_flat)
        
        # Total loss
        total_loss = self.weight_top * top_loss + self.weight_bottom * bottom_loss
        
        # Accuracy metrics
        top_acc = (predictions['top_indices'] == targets['top_indices']).float().mean()
        bottom_acc = (predictions['bottom_indices'] == targets['bottom_indices']).float().mean()
        
        return {
            'total_loss': total_loss,
            'top_loss': top_loss,
            'bottom_loss': bottom_loss,
            'top_accuracy': top_acc,
            'bottom_accuracy': bottom_acc
        }


def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Learn voltage to latent mapping")
    parser.add_argument("--voltage_dim", type=int, default=2, help="Number of voltage inputs")
    parser.add_argument("--top_size", type=int, default=32, help="Size of top matrix")
    parser.add_argument("--bottom_size", type=int, default=16, help="Size of bottom matrix")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--device_idx", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")

    # test the model on random data
    model = VoltagePixelCNN(
        voltage_dim=args.voltage_dim,
        top_matrix_size=args.top_size,
        bottom_matrix_size=args.bottom_size,
        embedding_dim=args.embedding_dim
    ).to(device)

    inputs = torch.randn(4, args.voltage_dim).to(device)  # Batch of 4 samples
    outputs = model(inputs)
    print("Top indices shape:", outputs['top_indices'].shape)
    print("Bottom indices shape:", outputs['bottom_indices'].shape)

if __name__ == "__main__":
    main()