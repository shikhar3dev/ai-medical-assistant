"""Neural Network Models for Disease Prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DiseasePredictor(nn.Module):
    """
    Multi-layer perceptron for disease risk prediction.
    
    Args:
        input_size: Number of input features
        hidden_layers: List of hidden layer sizes
        output_size: Number of output classes (1 for binary classification)
        dropout: Dropout rate for regularization
        batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'tanh', 'sigmoid')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [64, 32, 16],
        output_size: int = 1,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu"
    ):
        super(DiseasePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm and len(self.batch_norms) > i:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for binary classification.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for class imbalance.
    
    Args:
        pos_weight: Weight for positive class
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Weighted BCE loss value
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=inputs.device)
            return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets)


def get_loss_function(loss_type: str = "bce", **kwargs):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: Type of loss ('bce', 'focal', 'weighted_bce')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "weighted_bce":
        return WeightedBCELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_optimizer(model: nn.Module, optimizer_type: str = "adam", lr: float = 0.001, **kwargs):
    """
    Factory function to get optimizer.
    
    Args:
        model: Neural network model
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        lr: Learning rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_lr_scheduler(optimizer, scheduler_type: str = "step", **kwargs):
    """
    Factory function to get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('step', 'cosine', 'exponential', 'none')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
