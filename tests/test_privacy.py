"""Tests for Privacy Components"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from privacy.differential_privacy import DPTrainer, PrivacyEngine, PrivacyBudgetTracker
from federated.models import DiseasePredictor


class TestDPTrainer:
    """Test Differential Privacy Trainer."""
    
    def test_dp_trainer_creation(self):
        """Test DP trainer can be created."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        optimizer = torch.optim.Adam(model.parameters())
        
        dp_trainer = DPTrainer(
            model=model,
            optimizer=optimizer,
            max_grad_norm=1.0,
            noise_multiplier=1.1,
            epsilon=1.0,
            delta=1e-5
        )
        
        assert dp_trainer.max_grad_norm == 1.0
        assert dp_trainer.epsilon == 1.0
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        optimizer = torch.optim.Adam(model.parameters())
        
        dp_trainer = DPTrainer(
            model=model,
            optimizer=optimizer,
            max_grad_norm=1.0
        )
        
        # Create dummy gradients
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5, 1)).float()
        
        output = model(x)
        loss = nn.BCEWithLogitsLoss()(output, y)
        loss.backward()
        
        # Clip gradients
        grad_norm = dp_trainer.clip_gradients()
        
        assert grad_norm >= 0


class TestPrivacyEngine:
    """Test Privacy Engine."""
    
    def test_privacy_engine_creation(self):
        """Test privacy engine can be created."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        
        engine = PrivacyEngine(
            model=model,
            batch_size=32,
            sample_size=1000,
            epochs=10,
            epsilon=1.0,
            delta=1e-5
        )
        
        assert engine.epsilon == 1.0
        assert engine.noise_multiplier > 0


class TestPrivacyBudgetTracker:
    """Test Privacy Budget Tracker."""
    
    def test_tracker_creation(self):
        """Test tracker can be created."""
        tracker = PrivacyBudgetTracker(
            total_epsilon=1.0,
            total_delta=1e-5,
            num_rounds=50
        )
        
        assert tracker.total_epsilon == 1.0
        assert tracker.num_rounds == 50
    
    def test_budget_update(self):
        """Test budget update."""
        tracker = PrivacyBudgetTracker(
            total_epsilon=1.0,
            total_delta=1e-5,
            num_rounds=50
        )
        
        tracker.update(10)
        assert tracker.epsilon_spent == 10 * (1.0 / 50)
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion check."""
        tracker = PrivacyBudgetTracker(
            total_epsilon=1.0,
            total_delta=1e-5,
            num_rounds=50
        )
        
        tracker.update(48)
        assert tracker.is_budget_exhausted(threshold=0.95)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
