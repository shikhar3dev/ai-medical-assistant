"""Tests for Model Components."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from federated.models import DiseasePredictor, FocalLoss, get_loss_function, get_optimizer


class TestDiseasePredictor:
    """Test DiseasePredictor model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = DiseasePredictor(
            input_size=10,
            hidden_layers=[32, 16],
            output_size=1
        )
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        x = torch.randn(5, 10)
        output = model(x)
        
        assert output.shape == (5, 1)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        x = torch.randn(5, 10)
        probs = model.predict_proba(x)
        
        assert probs.shape == (5, 1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_predict(self):
        """Test class prediction."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        x = torch.randn(5, 10)
        preds = model.predict(x)
        
        assert preds.shape == (5, 1)
        assert torch.all((preds == 0) | (preds == 1))


class TestLossFunctions:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss."""
        loss_fn = FocalLoss()
        inputs = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()
        
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0
    
    def test_get_loss_function(self):
        """Test loss function factory."""
        bce_loss = get_loss_function('bce')
        assert isinstance(bce_loss, nn.BCEWithLogitsLoss)
        
        focal_loss = get_loss_function('focal')
        assert isinstance(focal_loss, FocalLoss)


class TestOptimizers:
    """Test optimizer creation."""
    
    def test_get_optimizer(self):
        """Test optimizer factory."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        
        adam = get_optimizer(model, 'adam', lr=0.001)
        assert isinstance(adam, torch.optim.Adam)
        
        sgd = get_optimizer(model, 'sgd', lr=0.001)
        assert isinstance(sgd, torch.optim.SGD)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
