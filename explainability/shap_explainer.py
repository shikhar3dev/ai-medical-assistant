"""SHAP-based Explainability for Disease Prediction Models"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import yaml
import pickle

from federated.models import DiseasePredictor
from preprocessing.preprocessor import DataPreprocessor


class SHAPExplainer:
    """
    SHAP explainer for disease prediction models.
    
    Args:
        model: Trained neural network model
        background_data: Background dataset for SHAP
        feature_names: List of feature names
        explainer_type: Type of SHAP explainer ('deep', 'gradient', 'kernel')
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        feature_names: List[str],
        explainer_type: str = 'deep'
    ):
        self.model = model
        self.model.eval()
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        
        # Convert background data to tensor
        self.background_tensor = torch.FloatTensor(background_data)
        
        # Create SHAP explainer
        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(model, self.background_tensor)
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(model, self.background_tensor)
        elif explainer_type == 'kernel':
            # Kernel explainer needs a prediction function
            def predict_fn(x):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    output = model(x_tensor)
                    probs = torch.sigmoid(output)
                return probs.numpy()
            
            self.explainer = shap.KernelExplainer(predict_fn, background_data)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        print(f"SHAP Explainer initialized ({explainer_type})")
        print(f"Background samples: {len(background_data)}")
        print(f"Features: {len(feature_names)}")
    
    def explain(self, X: np.ndarray) -> shap.Explanation:
        """
        Generate SHAP explanations for input data.
        
        Args:
            X: Input data to explain
            
        Returns:
            SHAP explanation object
        """
        if self.explainer_type in ['deep', 'gradient']:
            X_tensor = torch.FloatTensor(X)
            shap_values = self.explainer.shap_values(X_tensor)
        else:
            shap_values = self.explainer.shap_values(X)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X,
            feature_names=self.feature_names
        )
        
        return explanation
    
    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Get global feature importance.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame with feature importance scores
        """
        explanation = self.explain(X)
        shap_values = explanation.values
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def explain_instance(self, x: np.ndarray, instance_idx: int = 0) -> Dict:
        """
        Explain a single instance.
        
        Args:
            x: Input instance (can be 1D or 2D)
            instance_idx: Index of instance if x is 2D
            
        Returns:
            Dictionary with explanation details
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Get SHAP values
        explanation = self.explain(x)
        shap_values = explanation.values[instance_idx]
        
        # Get prediction
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            output = self.model(x_tensor)
            prob = torch.sigmoid(output).item()
        
        # Create feature contributions
        contributions = []
        for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            contributions.append({
                'feature': feature,
                'value': x[instance_idx, i],
                'shap_value': shap_val,
                'contribution': 'positive' if shap_val > 0 else 'negative'
            })
        
        # Sort by absolute SHAP value
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'prediction': prob,
            'predicted_class': 1 if prob >= 0.5 else 0,
            'contributions': contributions,
            'base_value': explanation.base_values if hasattr(explanation, 'base_values') else 0
        }
    
    def get_top_features(self, X: np.ndarray, top_k: int = 10) -> List[str]:
        """
        Get top K most important features.
        
        Args:
            X: Input data
            top_k: Number of top features to return
            
        Returns:
            List of top feature names
        """
        importance_df = self.get_feature_importance(X)
        return importance_df.head(top_k)['feature'].tolist()
    
    def save_explanations(self, X: np.ndarray, save_path: str) -> None:
        """
        Save explanations to file.
        
        Args:
            X: Input data
            save_path: Path to save explanations
        """
        explanation = self.explain(X)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        with open(save_path, 'wb') as f:
            pickle.dump(explanation, f)
        
        print(f"Explanations saved to {save_path}")


def load_model_and_data(config: Dict) -> Tuple[nn.Module, np.ndarray, np.ndarray, List[str]]:
    """
    Load trained model and test data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, test_data, test_labels, feature_names)
    """
    # Load preprocessor to get feature names
    preprocessor_path = Path(config['paths']['processed_data_dir']) / "preprocessor.pkl"
    preprocessor = DataPreprocessor.load(preprocessor_path)
    feature_names = preprocessor.feature_names
    
    # Load test data
    test_path = Path(config['paths']['processed_data_dir']) / "test_data.pt"
    test_data = torch.load(test_path)
    X_test = test_data['X'].numpy()
    y_test = test_data['y'].numpy()
    
    # Load model
    model_path = Path(config['paths']['models_dir']) / "best_model.pt"
    checkpoint = torch.load(model_path)
    
    # Create model
    input_size = X_test.shape[1]
    model = DiseasePredictor(
        input_size=input_size,
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout'],
        batch_norm=config['model']['batch_norm'],
        activation=config['model']['activation']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Test data: {X_test.shape}")
    
    return model, X_test, y_test, feature_names


def main():
    """Main function for SHAP explanation generation."""
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument("--config", type=str, default="configs/fl_config.yaml", help="Config file")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to explain")
    parser.add_argument("--explainer-type", type=str, default="deep", 
                       choices=["deep", "gradient", "kernel"], help="SHAP explainer type")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and data
    print("Loading model and data...")
    model, X_test, y_test, feature_names = load_model_and_data(config)
    
    # Sample background data
    background_samples = config['explainability']['shap_background_samples']
    background_indices = np.random.choice(len(X_test), size=min(background_samples, len(X_test)), replace=False)
    background_data = X_test[background_indices]
    
    # Create SHAP explainer
    print("\nCreating SHAP explainer...")
    explainer = SHAPExplainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names,
        explainer_type=args.explainer_type
    )
    
    # Generate explanations
    print("\nGenerating explanations...")
    num_samples = min(args.num_samples, len(X_test))
    X_explain = X_test[:num_samples]
    
    # Get feature importance
    print("\nCalculating feature importance...")
    importance_df = explainer.get_feature_importance(X_explain)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Save feature importance
    importance_path = Path(config['paths']['explanations_dir']) / "feature_importance.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to {importance_path}")
    
    # Explain a few instances
    print("\nExplaining sample instances...")
    for i in range(min(5, num_samples)):
        instance_explanation = explainer.explain_instance(X_explain, i)
        print(f"\nInstance {i}:")
        print(f"  Prediction: {instance_explanation['prediction']:.4f}")
        print(f"  Predicted class: {instance_explanation['predicted_class']}")
        print(f"  Top 5 contributing features:")
        for contrib in instance_explanation['contributions'][:5]:
            print(f"    {contrib['feature']}: {contrib['shap_value']:.4f} ({contrib['contribution']})")
    
    # Save explanations
    explanations_path = Path(config['paths']['explanations_dir']) / "shap_explanations.pkl"
    explainer.save_explanations(X_explain, explanations_path)
    
    print("\nSHAP explanation generation complete!")


if __name__ == "__main__":
    main()
