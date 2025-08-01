import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import math


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of vector quantization.
        
        Args:
            inputs: Tensor of shape (batch_size, embedding_dim, height, width)
            
        Returns:
            quantized: Quantized tensor
            vq_loss: Vector quantization loss
            perplexity: Perplexity measure
            encoding_indices: Discrete indices of the embeddings
        """
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), vq_loss, perplexity, encoding_indices.view(input_shape[:-1])


class ResidualBlock(nn.Module):
    """Residual block with groupnorm and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()
        
        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_connection(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        return self.activation(out + residual)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 8, groups: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(1)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return out + residual


class DownsampleBlock(nn.Module):
    """Downsampling block with residual connections and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2, 
                 downsample: bool = True, use_attention: bool = False):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])
        
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x)
            
        if self.attention is not None:
            x = self.attention(x)
            
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block with residual connections and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2, 
                 upsample: bool = True, use_attention: bool = False):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1) if upsample else None
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])
        
        self.attention = AttentionBlock(out_channels) if use_attention else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
            
        for res_block in self.res_blocks:
            x = res_block(x)
            
        if self.attention is not None:
            x = self.attention(x)
            
        return x


class Encoder(nn.Module):
    """
    Hierarchical encoder for VQ-VAE 2.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 128, 
                 channel_multipliers: List[int] = [1, 1, 2, 2, 4, 4], 
                 num_res_blocks: int = 2, attention_resolutions: List[int] = [16, 8]):
        super().__init__()
        
        self.num_resolutions = len(channel_multipliers)
        self.attention_resolutions = attention_resolutions
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            downsample = i < self.num_resolutions - 1
            
            # Calculate current resolution (assuming input is 256x256)
            current_res = 256 // (2 ** i)
            use_attention = current_res in attention_resolutions
            
            self.down_blocks.append(
                DownsampleBlock(in_ch, out_ch, num_res_blocks, downsample, use_attention)
            )
            in_ch = out_ch
        
        # Middle blocks
        self.mid_block1 = ResidualBlock(in_ch, in_ch)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch)
        
        # Output layers for different levels
        self.norm_out_top = nn.GroupNorm(32, in_ch)
        self.conv_out_top = nn.Conv2d(in_ch, 512, 3, padding=1)  # Top level codebook
        
        # For bottom level, we need to upsample once
        self.upsample_bottom = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.norm_out_bottom = nn.GroupNorm(32, in_ch)
        self.conv_out_bottom = nn.Conv2d(in_ch, 256, 3, padding=1)  # Bottom level codebook
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning encodings for both levels.
        
        Returns:
            top_encoding: Encoding for top level (lower resolution)
            bottom_encoding: Encoding for bottom level (higher resolution)
        """
        x = self.conv_in(x)
        
        # Store intermediate features
        features = []
        for down_block in self.down_blocks:
            x = down_block(x)
            features.append(x)
        
        # Middle processing
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        # Top level encoding (lowest resolution)
        top_encoding = self.norm_out_top(x)
        top_encoding = F.silu(top_encoding)
        top_encoding = self.conv_out_top(top_encoding)
        
        # Bottom level encoding (higher resolution)
        bottom_features = self.upsample_bottom(x)
        bottom_encoding = self.norm_out_bottom(bottom_features)
        bottom_encoding = F.silu(bottom_encoding)
        bottom_encoding = self.conv_out_bottom(bottom_encoding)
        
        return top_encoding, bottom_encoding


class Decoder(nn.Module):
    """
    Hierarchical decoder for VQ-VAE 2.
    """
    def __init__(self, out_channels: int = 3, base_channels: int = 128,
                 channel_multipliers: List[int] = [4, 4, 2, 2, 1, 1],
                 num_res_blocks: int = 2, attention_resolutions: List[int] = [16, 8],
                 input_resolution: int = 64):
        super().__init__()
        
        self.num_resolutions = len(channel_multipliers)
        self.attention_resolutions = attention_resolutions
        self.input_resolution = input_resolution
        
        # Calculate how many downsampling steps the encoder does
        # The encoder downsamples in all blocks except the last one
        self.num_downsamples = self.num_resolutions - 1
        self.encoded_resolution = input_resolution // (2 ** self.num_downsamples)
        
        # Input layers for different levels
        self.conv_in_top = nn.Conv2d(512, base_channels * channel_multipliers[0], 3, padding=1)
        self.conv_in_bottom = nn.Conv2d(256, base_channels * channel_multipliers[0], 3, padding=1)
        
        # Combine top and bottom features
        self.combine_conv = nn.Conv2d(base_channels * channel_multipliers[0] * 2, 
                                     base_channels * channel_multipliers[0], 3, padding=1)
        
        # Middle blocks
        in_ch = base_channels * channel_multipliers[0]
        self.mid_block1 = ResidualBlock(in_ch, in_ch)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch)
        
        # Upsampling blocks - exactly match the number of downsamples
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            # Only upsample for the first num_downsamples blocks
            upsample = i < self.num_downsamples
            
            # Calculate current resolution during upsampling
            current_res = self.encoded_resolution * (2 ** min(i, self.num_downsamples))
            use_attention = current_res in attention_resolutions
            
            self.up_blocks.append(
                UpsampleBlock(in_ch, out_ch, num_res_blocks, upsample, use_attention)
            )
            in_ch = out_ch
        
        # Output layer
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)
        
    def forward(self, top_quantized: torch.Tensor, bottom_quantized: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining both quantized levels.
        
        Args:
            top_quantized: Quantized features from top level
            bottom_quantized: Quantized features from bottom level
            
        Returns:
            Reconstructed image
        """
        # Process top level
        top_features = self.conv_in_top(top_quantized)
        
        # Upsample top features to match bottom resolution
        top_features = F.interpolate(top_features, size=bottom_quantized.shape[-2:], mode='nearest')
        
        # Process bottom level
        bottom_features = self.conv_in_bottom(bottom_quantized)
        
        # Combine features
        combined = torch.cat([top_features, bottom_features], dim=1)
        x = self.combine_conv(combined)
        
        # Middle processing
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        # Upsampling
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        # Ensure output matches input resolution exactly
        if x.shape[-2:] != (self.input_resolution, self.input_resolution):
            x = F.interpolate(x, size=(self.input_resolution, self.input_resolution), 
                            mode='bilinear', align_corners=False)
        
        return x


class VQVAE2(nn.Module):
    """
    VQ-VAE 2 model for high-quality image reconstruction.
    """
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 128,
                 num_embeddings_top: int = 512,
                 num_embeddings_bottom: int = 512,
                 embedding_dim_top: int = 512,
                 embedding_dim_bottom: int = 256,
                 commitment_cost: float = 0.25,
                 channel_multipliers: List[int] = [1, 1, 2, 2, 4, 4],
                 num_res_blocks: int = 2,
                 attention_resolutions: List[int] = [16, 8],
                 input_resolution: int = 64):  # Add input resolution parameter
        super().__init__()
        
        self.input_resolution = input_resolution
        
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions
        )
        
        self.decoder = Decoder(
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=list(reversed(channel_multipliers)),
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            input_resolution=input_resolution  # Pass input resolution to decoder
        )
        
        self.vq_top = VectorQuantizer(
            num_embeddings=num_embeddings_top,
            embedding_dim=embedding_dim_top,
            commitment_cost=commitment_cost
        )
        
        self.vq_bottom = VectorQuantizer(
            num_embeddings=num_embeddings_bottom,
            embedding_dim=embedding_dim_bottom,
            commitment_cost=commitment_cost
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of VQ-VAE 2.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed images
                - vq_loss_top: VQ loss for top level
                - vq_loss_bottom: VQ loss for bottom level
                - perplexity_top: Perplexity for top level
                - perplexity_bottom: Perplexity for bottom level
        """
        # Encode
        top_encoding, bottom_encoding = self.encoder(x)

        # Quantize
        top_quantized, vq_loss_top, perplexity_top, top_indices = self.vq_top(top_encoding)
        bottom_quantized, vq_loss_bottom, perplexity_bottom, bottom_indices = self.vq_bottom(bottom_encoding)

        # Decode
        reconstructed = self.decoder(top_quantized, bottom_quantized)

        return {
            'reconstructed': reconstructed,
            'vq_loss_top': vq_loss_top,
            'vq_loss_bottom': vq_loss_bottom,
            'perplexity_top': perplexity_top,
            'perplexity_bottom': perplexity_bottom,
            'top_quantized': top_quantized,
            'bottom_quantized': bottom_quantized,
            'top_indices': top_indices,
            'bottom_indices': bottom_indices
        }
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to quantized representations."""
        top_encoding, bottom_encoding = self.encoder(x)
        top_quantized, _, _ = self.vq_top(top_encoding)
        bottom_quantized, _, _ = self.vq_bottom(bottom_encoding)
        return top_quantized, bottom_quantized

    def encode_to_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to quantization indices.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of top and bottom quantization indices.
        """
        top_encoding, bottom_encoding = self.encoder(x)
        top_indices = self.vq_top(top_encoding)[3]
        bottom_indices = self.vq_bottom(bottom_encoding)[3]
        return top_indices, bottom_indices

    def decode(self, top_quantized: torch.Tensor, bottom_quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized representations to images."""
        return self.decoder(top_quantized, bottom_quantized)

    def decode_from_indices(self, top_indices: torch.Tensor, bottom_indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the image from quantization indices.

        Args:
            top_indices: Tensor of shape (batch_size, height_top, width_top) containing top-level quantization indices.
            bottom_indices: Tensor of shape (batch_size, height_bottom, width_bottom) containing bottom-level quantization indices.

        Returns:
            Reconstructed image as a tensor of shape (batch_size, channels, height, width).
        """
        # Retrieve embeddings from the codebook using the indices
        top_quantized = self.vq_top.embeddings(top_indices.view(-1)).view(
            top_indices.shape[0], top_indices.shape[1], top_indices.shape[2], -1
        ).permute(0, 3, 1, 2)  # Reshape to (batch_size, embedding_dim, height, width)

        bottom_quantized = self.vq_bottom.embeddings(bottom_indices.view(-1)).view(
            bottom_indices.shape[0], bottom_indices.shape[1], bottom_indices.shape[2], -1
        ).permute(0, 3, 1, 2)  # Reshape to (batch_size, embedding_dim, height, width)

        # Decode the quantized embeddings
        reconstructed = self.decoder(top_quantized, bottom_quantized)
        return reconstructed


def create_vqvae2_large(image_size: int = 256, in_channels: int = 3) -> VQVAE2:
    """
    Create a large VQ-VAE 2 model suitable for complex image datasets.
    
    Args:
        image_size: Input image size (assumed square)
        in_channels: Number of input channels
        
    Returns:
        Configured VQ-VAE 2 model
    """
    # Scale model based on image size
    if image_size <= 64:
        base_channels = 64
        channel_multipliers = [1, 2, 4]  # Fewer layers for smaller images
        attention_resolutions = []
    elif image_size <= 128:
        base_channels = 64
        channel_multipliers = [1, 2, 2, 4]
        attention_resolutions = [16, 8]
    elif image_size <= 256:
        base_channels = 128
        channel_multipliers = [1, 1, 2, 2, 4, 4]
        attention_resolutions = [32, 16, 8]
    else:  # 512 or larger
        base_channels = 128
        channel_multipliers = [1, 1, 2, 2, 4, 4, 8]
        attention_resolutions = [64, 32, 16, 8]
    
    return VQVAE2(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        num_embeddings_top=1024,
        num_embeddings_bottom=1024,
        embedding_dim_top=512,
        embedding_dim_bottom=256,
        commitment_cost=0.25,
        channel_multipliers=channel_multipliers,
        num_res_blocks=2,
        attention_resolutions=attention_resolutions,
        input_resolution=image_size  # Pass the input resolution
    )


# Training utilities
class VQVAELoss(nn.Module):
    """Combined loss for VQ-VAE 2 training with latent consistency."""
    
    def __init__(self, reconstruction_weight: float = 1.0, vq_weight: float = 1.0, 
                 consistency_weight: float = 0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.vq_weight = vq_weight
        self.consistency_weight = consistency_weight
        
    def compute_consistency_loss(self, latents: torch.Tensor, voltages: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between voltage distances and latent distances.
        
        Args:
            latents: Latent representations [batch_size, embedding_dim, h, w]
            voltages: Voltage parameters [batch_size, num_voltages]
            
        Returns:
            Consistency loss scalar
        """
        batch_size = latents.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=latents.device)
        
        # Flatten latents to [batch_size, -1]
        latents_flat = latents.view(batch_size, -1)
        
        # Compute pairwise distances in voltage space
        voltage_diffs = voltages.unsqueeze(1) - voltages.unsqueeze(0)  # [B, B, num_voltages]
        voltage_distances = torch.norm(voltage_diffs, dim=2)  # [B, B]
        
        # Compute pairwise distances in latent space
        latent_diffs = latents_flat.unsqueeze(1) - latents_flat.unsqueeze(0)  # [B, B, latent_dim]
        latent_distances = torch.norm(latent_diffs, dim=2)  # [B, B]
        
        # Create masks to exclude diagonal (same sample comparisons)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=latents.device)
        
        # Extract upper triangular part to avoid double counting
        triu_mask = torch.triu(torch.ones_like(mask), diagonal=1)
        final_mask = mask & triu_mask
        
        voltage_dist_pairs = voltage_distances[final_mask]
        latent_dist_pairs = latent_distances[final_mask]
        
        if len(voltage_dist_pairs) == 0:
            return torch.tensor(0.0, device=latents.device)
        
        # Normalize distances to [0, 1] range for stability
        voltage_dist_norm = voltage_dist_pairs / (voltage_dist_pairs.max() + 1e-8)
        latent_dist_norm = latent_dist_pairs / (latent_dist_pairs.max() + 1e-8)
        
        # Compute consistency loss using MSE between normalized distances
        consistency_loss = F.mse_loss(latent_dist_norm, voltage_dist_norm)
        
        return consistency_loss
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, 
                voltages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VQ-VAE 2 loss with consistency term.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target images
            voltages: Voltage parameters [batch_size, num_voltages]
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstructed'], targets)
        
        # VQ losses
        vq_loss = outputs['vq_loss_top'] + outputs['vq_loss_bottom']
        
        # Consistency loss using top-level quantized features
        consistency_loss = self.compute_consistency_loss(
            outputs['top_quantized'], voltages
        )
        
        # Total loss
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.vq_weight * vq_loss + 
                     self.consistency_weight * consistency_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'vq_loss': vq_loss,
            'consistency_loss': consistency_loss,
            'vq_loss_top': outputs['vq_loss_top'],
            'vq_loss_bottom': outputs['vq_loss_bottom'],
            'perplexity_top': outputs['perplexity_top'],
            'perplexity_bottom': outputs['perplexity_bottom']
        }


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_vqvae2_large(image_size=128, in_channels=1)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    
    print(f"Input shape: {x.shape}")

    output = model(x)
    tq, bq = output['top_indices'], output['bottom_indices']
    print(tq.shape)
    print(bq.shape)
    import sys
    sys.exit(0)

    tq, bq = model.encode_to_indices(x)
    print(tq.shape)
    print(bq.shape)
    import sys
    sys.exit(0)
    
    # Forward pass
    outputs = model(x)
    
    print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"VQ loss top: {outputs['vq_loss_top'].item():.4f}")
    print(f"VQ loss bottom: {outputs['vq_loss_bottom'].item():.4f}")
    print(f"Perplexity top: {outputs['perplexity_top'].item():.2f}")
    print(f"Perplexity bottom: {outputs['perplexity_bottom'].item():.2f}")
    
    # Test loss computation
    loss_fn = VQVAELoss()
    losses = loss_fn(outputs, x)
    
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")