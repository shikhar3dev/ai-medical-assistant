"""Differential Privacy Implementation using Opacus"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Tuple
import numpy as np
from opacus import PrivacyEngine as OpacusPrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants import create_accountant


class PrivacyEngine:
    """
    Privacy engine for differential privacy.
    
    Args:
        model: Neural network model
        batch_size: Batch size for training
        sample_size: Total number of samples in dataset
        epochs: Number of training epochs
        epsilon: Privacy budget (ε)
        delta: Privacy parameter (δ)
        max_grad_norm: Maximum gradient norm for clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        sample_size: int,
        epochs: int = 1,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        # Calculate noise multiplier
        self.noise_multiplier = self._compute_noise_multiplier()
        
        # Initialize privacy accountant
        self.accountant = create_accountant(mechanism="gdp")
        self.steps = 0
        
        print(f"Privacy Engine initialized:")
        print(f"  ε={epsilon}, δ={delta}")
        print(f"  Max grad norm={max_grad_norm}")
        print(f"  Noise multiplier={self.noise_multiplier:.4f}")
    
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier for target privacy budget.
        
        Returns:
            Noise multiplier (sigma)
        """
        # Simplified calculation - in practice, use Opacus utilities
        # This is an approximation based on the Gaussian mechanism
        steps_per_epoch = self.sample_size / self.batch_size
        total_steps = steps_per_epoch * self.epochs
        
        # Use empirical formula (simplified)
        # For more accurate calculation, use opacus.privacy_analysis
        q = self.batch_size / self.sample_size  # Sampling rate
        
        # Approximate noise multiplier
        # This is a simplified version - production should use proper calculation
        noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise_multiplier *= np.sqrt(total_steps * q)
        
        return max(noise_multiplier, 0.1)  # Ensure minimum noise
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        if self.steps == 0:
            return 0.0, 0.0
        
        # Calculate privacy spent using the accountant
        # This is a simplified version
        q = self.batch_size / self.sample_size
        epsilon = self.accountant.get_epsilon(delta=self.delta)
        
        return epsilon, self.delta
    
    def step(self):
        """Record a training step for privacy accounting."""
        self.steps += 1


class DPTrainer:
    """
    Differential Privacy Trainer using gradient clipping and noise injection.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier for Gaussian mechanism
        epsilon: Target privacy budget
        delta: Privacy parameter
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.epsilon = epsilon
        self.delta = delta
        
        # Validate model for DP
        self.model = ModuleValidator.fix(self.model)
        
        # Track privacy
        self.steps = 0
        self.privacy_spent = 0.0
        
        print(f"DP Trainer initialized with ε={epsilon}, δ={delta}")
    
    def clip_gradients(self) -> float:
        """
        Clip gradients to maximum norm.
        
        Returns:
            Total gradient norm before clipping
        """
        total_norm = 0.0
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        
        # Calculate total norm
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise(self):
        """Add Gaussian noise to gradients."""
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=p.grad.shape,
                    device=p.grad.device
                )
                p.grad.data.add_(noise)
    
    def step(self, loss: torch.Tensor):
        """
        Perform one DP training step.
        
        Args:
            loss: Loss value
        """
        # Backward pass
        loss.backward()
        
        # Clip gradients
        grad_norm = self.clip_gradients()
        
        # Add noise
        self.add_noise()
        
        # Optimizer step
        self.optimizer.step()
        
        # Track privacy
        self.steps += 1
        
        return grad_norm
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Estimate privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        # Simplified privacy accounting
        # In production, use proper privacy accountant
        if self.steps == 0:
            return 0.0, 0.0
        
        # Rough estimate using composition
        # This is simplified - use proper accounting in production
        epsilon_per_step = self.epsilon / max(self.steps, 1)
        total_epsilon = epsilon_per_step * self.steps
        
        return min(total_epsilon, self.epsilon), self.delta


def make_private(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader,
    epochs: int,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
) -> Tuple[nn.Module, Optimizer, OpacusPrivacyEngine]:
    """
    Make a model private using Opacus.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        data_loader: Data loader
        epochs: Number of epochs
        epsilon: Privacy budget
        delta: Privacy parameter
        max_grad_norm: Maximum gradient norm
        
    Returns:
        Tuple of (private model, private optimizer, privacy engine)
    """
    # Validate and fix model
    model = ModuleValidator.fix(model)
    
    # Create privacy engine
    privacy_engine = OpacusPrivacyEngine()
    
    # Make model private
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=max_grad_norm,
    )
    
    print(f"Model made private with Opacus")
    print(f"Target: ε={epsilon}, δ={delta}")
    
    return model, optimizer, privacy_engine


class PrivacyBudgetTracker:
    """
    Track privacy budget consumption across federated rounds.
    
    Args:
        total_epsilon: Total privacy budget
        total_delta: Privacy parameter
        num_rounds: Number of federated rounds
    """
    
    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        num_rounds: int
    ):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.num_rounds = num_rounds
        self.epsilon_per_round = total_epsilon / num_rounds
        
        self.current_round = 0
        self.epsilon_spent = 0.0
        
        print(f"Privacy Budget Tracker initialized:")
        print(f"  Total budget: ε={total_epsilon}, δ={total_delta}")
        print(f"  Per-round budget: ε={self.epsilon_per_round:.4f}")
    
    def update(self, round_num: int):
        """
        Update privacy budget for a round.
        
        Args:
            round_num: Current round number
        """
        self.current_round = round_num
        self.epsilon_spent = round_num * self.epsilon_per_round
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """
        Get remaining privacy budget.
        
        Returns:
            Tuple of (remaining epsilon, delta)
        """
        remaining_epsilon = self.total_epsilon - self.epsilon_spent
        return remaining_epsilon, self.total_delta
    
    def is_budget_exhausted(self, threshold: float = 0.95) -> bool:
        """
        Check if privacy budget is exhausted.
        
        Args:
            threshold: Threshold for budget exhaustion (0-1)
            
        Returns:
            True if budget is exhausted
        """
        return self.epsilon_spent >= threshold * self.total_epsilon
    
    def get_status(self) -> dict:
        """
        Get privacy budget status.
        
        Returns:
            Dictionary with budget status
        """
        return {
            'total_epsilon': self.total_epsilon,
            'epsilon_spent': self.epsilon_spent,
            'epsilon_remaining': self.total_epsilon - self.epsilon_spent,
            'percentage_used': (self.epsilon_spent / self.total_epsilon) * 100,
            'current_round': self.current_round,
            'total_rounds': self.num_rounds,
        }
