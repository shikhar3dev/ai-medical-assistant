"""Tests for Explainability Components"""

import pytest
import torch
import numpy as np
import pandas as pd

from federated.models import DiseasePredictor
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer


class TestSHAPExplainer:
    """Test SHAP Explainer."""
    
    @pytest.fixture
    def setup_explainer(self):
        """Setup model and explainer."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        background_data = np.random.randn(50, 10)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        explainer = SHAPExplainer(
            model=model,
            background_data=background_data,
            feature_names=feature_names,
            explainer_type='deep'
        )
        
        return model, explainer, background_data, feature_names
    
    def test_explainer_creation(self, setup_explainer):
        """Test explainer can be created."""
        model, explainer, background_data, feature_names = setup_explainer
        
        assert explainer.model == model
        assert len(explainer.feature_names) == 10
    
    def test_get_feature_importance(self, setup_explainer):
        """Test feature importance calculation."""
        model, explainer, background_data, feature_names = setup_explainer
        
        X = np.random.randn(20, 10)
        importance_df = explainer.get_feature_importance(X)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == 10
    
    def test_explain_instance(self, setup_explainer):
        """Test instance explanation."""
        model, explainer, background_data, feature_names = setup_explainer
        
        x = np.random.randn(10)
        explanation = explainer.explain_instance(x)
        
        assert 'prediction' in explanation
        assert 'predicted_class' in explanation
        assert 'contributions' in explanation


class TestLIMEExplainer:
    """Test LIME Explainer."""
    
    @pytest.fixture
    def setup_explainer(self):
        """Setup model and explainer."""
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        training_data = np.random.randn(100, 10)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        explainer = LIMEExplainer(
            model=model,
            training_data=training_data,
            feature_names=feature_names,
            num_samples=1000
        )
        
        return model, explainer, training_data, feature_names
    
    def test_explainer_creation(self, setup_explainer):
        """Test explainer can be created."""
        model, explainer, training_data, feature_names = setup_explainer
        
        assert explainer.model == model
        assert len(explainer.feature_names) == 10
    
    def test_explain_instance_detailed(self, setup_explainer):
        """Test detailed instance explanation."""
        model, explainer, training_data, feature_names = setup_explainer
        
        x = np.random.randn(10)
        explanation = explainer.explain_instance_detailed(x, num_features=5)
        
        assert 'prediction_proba' in explanation
        assert 'predicted_class' in explanation
        assert 'contributions' in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
