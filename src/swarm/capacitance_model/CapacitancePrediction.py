import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim


# =============================================================================
# IMPALA CNN Backbone (lightweight, ~100-200K params)
# =============================================================================

class ResNetBlock(nn.Module):
    """
    IMPALA residual block: y = x + Conv3x3(ReLU(Conv3x3(ReLU(x))))
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = F.relu(x)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        return x + y


class IMPALABackbone(nn.Module):
    """
    IMPALA CNN encoder - lightweight backbone for capacitance prediction.

    Architecture: Conv -> MaxPool -> ResBlocks -> ReLU (repeated per stage)
    Default: 16->32->32 channels with 2 ResBlocks per stage (~100-200K params)
    """

    def __init__(self, in_channels: int = 1, channels: list = None, num_res_blocks: int = 2):
        super().__init__()

        if channels is None:
            channels = [16, 32, 32]  # Default IMPALA channel progression

        self.channels = channels
        self.num_res_blocks = num_res_blocks

        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(channels):
            # Conv layer
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1))
            # MaxPool for downsampling (stride 2)
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # ResNet blocks
            for _ in range(num_res_blocks):
                layers.append(ResNetBlock(out_channels))
            # ReLU after blocks
            layers.append(nn.ReLU())
            prev_channels = out_channels

        # Adaptive pooling to fixed size
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        layers.append(nn.Flatten())

        self.cnn = nn.Sequential(*layers)
        self.feature_dim = channels[-1] * 4 * 4  # After adaptive pooling

    def forward(self, x):
        return self.cnn(x)


class IMPALACapacitanceModel(nn.Module):
    """
    IMPALA-based model for capacitance prediction with uncertainty estimation.
    Much lighter than MobileNet (~100-200K params vs ~2.5M).
    """

    def __init__(self, output_size: int = 3, channels: list = None, num_res_blocks: int = 2):
        super().__init__()

        self.backbone = IMPALABackbone(in_channels=1, channels=channels, num_res_blocks=num_res_blocks)
        feature_dim = self.backbone.feature_dim
        self.output_size = output_size

        # Value head for capacitance predictions
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size),
        )

        # Confidence head for uncertainty estimation (log variance)
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        features = self.backbone(x)
        values = self.value_head(features)
        log_vars = self.confidence_head(features)
        return values, log_vars


class CapacitancePredictionModel(nn.Module):
    """
    ResNet18-based model for capacitance prediction with uncertainty estimation.
    
    Takes 2-channel images as input and outputs:
    - 3 continuous values (capacitance predictions)
    - 3 confidence scores for uncertainty estimation
    """
    
    def __init__(self, output_size, mobilenet="small"):
        super().__init__()
        
        # Load pretrained MobileNetV3 backbone
        if mobilenet == "small":
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            feature_dim = 576
        elif mobilenet == "large":
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            feature_dim = 960

        self.output_size = output_size

        # Modify first conv layer to accept 1 channels instead of 3
        # In MobileNetV3, the first conv layer is features[0][0]
        original_conv1 = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,  # Changed from 3 to 1 channels
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize new conv1 weights
        with torch.no_grad():
            # Use the first 1 channels of the original conv1 weights
            self.backbone.features[0][0].weight = nn.Parameter(
                original_conv1.weight[:, :1, :, :].clone()
            )
        
        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()
        
        # Custom prediction heads
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.output_size),
            #nn.Sigmoid()
        )
        
        # Confidence head (outputs log variance for uncertainty estimation)
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.output_size)
        )
    
    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 2, height, width)

        Returns:
            values: Predicted continuous values (batch_size, output_size)
            log_vars: Log variance for uncertainty (batch_size, output_size)
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)

        # Predict values and log variances
        values = self.value_head(features)
        log_vars = self.confidence_head(features)

        return values, log_vars


class CapacitanceLoss(nn.Module):
    """
    Combined loss function for capacitance prediction with uncertainty.
    
    Combines MSE loss for values with negative log-likelihood loss for confidence.
    """
    
    def __init__(self, mse_weight=1.0, nll_weight=1.0, beta=0.5):
        super(CapacitanceLoss, self).__init__()
        self.mse_weight = mse_weight
        self.nll_weight = nll_weight
        self.beta = beta


    def _beta_nll_loss(self, mean, logvar, target):
        """Beta NLL loss to improve variance prediction stability"""
        var = torch.exp(logvar)
        errors = (mean - target) ** 2

        nll = 0.5 * (logvar + errors / var)

        if self.beta > 0:
            weights = var.detach() ** self.beta
            nll = nll * weights
        
        return nll.mean(), errors

    
    def forward(self, predictions, targets):
        """
        Compute combined loss
        
        Args:
            predictions: Tuple of (values, log_vars) from model
            targets: Ground truth values (batch_size, 3)
            
        Returns:
            total_loss: Combined loss
            mse_loss: MSE component
            nll_loss: Negative log-likelihood component
        """
        values, log_vars = predictions
        
        # MSE loss for the predicted values
        mse_loss = F.mse_loss(values, targets)
        
        # Negative log-likelihood loss with predicted uncertainty
        nll_loss, squared_errors = self._beta_nll_loss(values, log_vars, targets)

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.nll_weight * nll_loss
        
        return total_loss, mse_loss, nll_loss, log_vars, squared_errors


def create_model(output_size, backbone="mobilenet", mobilenet="small", impala_channels=None, num_res_blocks=2):
    """
    Factory function to create the model.

    Args:
        output_size: Number of output values to predict
        backbone: "mobilenet" or "impala"
        mobilenet: "small" or "large" (only used if backbone="mobilenet")
        impala_channels: Channel list for IMPALA (default [16, 32, 32])
        num_res_blocks: Number of ResNet blocks per stage for IMPALA (default 2)
    """
    if backbone == "impala":
        return IMPALACapacitanceModel(output_size, channels=impala_channels, num_res_blocks=num_res_blocks)
    else:
        return CapacitancePredictionModel(output_size, mobilenet=mobilenet)


def create_loss_function(mse_weight=1.0, nll_weight=0.1):
    """Factory function to create the loss function"""
    return CapacitanceLoss(mse_weight, nll_weight)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model(output_size=2)
    loss_fn = create_loss_function()
    
    # Example input (batch_size=4, channels=2, height=224, width=224)
    batch_size = 4
    x = torch.randn(batch_size, 1, 100, 100)
    targets = torch.randn(batch_size, 2)  # Ground truth values
    
    # Forward pass
    values, log_vars = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predicted values shape: {values.shape}")
    print(f"Predicted log variances shape: {log_vars.shape}")
    
    # Compute loss
    total_loss, mse_loss, nll_loss, log_vars_out, squared_errors = loss_fn((values, log_vars), targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"MSE loss: {mse_loss.item():.4f}")
    print(f"NLL loss: {nll_loss.item():.4f}")
    
    # Training setup example
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("Training step completed successfully!")
    
    # Extract uncertainty (standard deviation) from log variance
    uncertainties = torch.exp(0.5 * log_vars)
    print(f"Predicted uncertainties (std): {uncertainties}")