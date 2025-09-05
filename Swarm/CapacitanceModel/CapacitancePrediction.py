import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim


class CapacitancePredictionModel(nn.Module):
    """
    ResNet18-based model for capacitance prediction with uncertainty estimation.
    
    Takes 2-channel images as input and outputs:
    - 3 continuous values (capacitance predictions)
    - 3 confidence scores for uncertainty estimation
    """
    
    def __init__(self):
        super().__init__()
        
        # Load pretrained MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
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
        
        # Get the feature dimension (MobileNetV3-small outputs 576 features)
        feature_dim = 576
        
        # Custom prediction heads
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3), # 3 continuous values
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
            nn.Linear(128, 3)  # 3 log variance values
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 2, height, width)
            
        Returns:
            values: Predicted continuous values (batch_size, 3)
            log_vars: Log variance for uncertainty (batch_size, 3)
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
    
    def __init__(self, mse_weight=1.0, nll_weight=1.0):
        super(CapacitanceLoss, self).__init__()
        self.mse_weight = mse_weight
        self.nll_weight = nll_weight
    
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
        # NLL = 0.5 * (log(2π) + log_var + (target - pred)^2 / exp(log_var))
        variances = torch.exp(log_vars)
        squared_errors = (targets - values) ** 2
        
        # nll_loss = 0.5 * (
        #     torch.log(2 * torch.pi * variances) + 
        #     squared_errors / variances
        # ).mean()

        nll_loss = 0.5 * (
            log_vars + 
            squared_errors / variances
        ).mean()

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.nll_weight * nll_loss
        
        return total_loss, mse_loss, nll_loss, log_vars, squared_errors


def create_model():
    """Factory function to create the model"""
    return CapacitancePredictionModel()


def create_loss_function(mse_weight=1.0, nll_weight=0.1):
    """Factory function to create the loss function"""
    return CapacitanceLoss(mse_weight, nll_weight)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model()
    loss_fn = create_loss_function()
    
    # Example input (batch_size=4, channels=2, height=224, width=224)
    batch_size = 4
    x = torch.randn(batch_size, 1, 224, 224)
    targets = torch.randn(batch_size, 3)  # Ground truth values
    
    # Forward pass
    values, log_vars = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predicted values shape: {values.shape}")
    print(f"Predicted log variances shape: {log_vars.shape}")
    
    # Compute loss
    total_loss, mse_loss, nll_loss = loss_fn((values, log_vars), targets)
    
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
