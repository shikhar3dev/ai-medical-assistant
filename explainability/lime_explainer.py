"""LIME-based Explainability for Disease Prediction Models"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lime import lime_tabular
import matplotlib.pyplot as plt
import yaml
import pickle

from federated.models import DiseasePredictor
from preprocessing.preprocessor import DataPreprocessor


class LIMEExplainer:
    """
    LIME explainer for disease prediction models.
    
    Args:
        model: Trained neural network model
        training_data: Training data for LIME
        feature_names: List of feature names
        class_names: List of class names
        num_samples: Number of samples for LIME
    """
    
    def __init__(
        self,
        model: nn.Module,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str] = ['No Disease', 'Disease'],
        num_samples: int = 5000
    ):
        self.model = model
        self.model.eval()
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.num_samples = num_samples
        
        # Create prediction function for LIME
        def predict_fn(X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                output = self.model(X_tensor)
                probs = torch.sigmoid(output).numpy()
                # Return probabilities for both classes
                return np.hstack([1 - probs, probs])
        
        self.predict_fn = predict_fn
        
        # Create LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        print(f"LIME Explainer initialized")
        print(f"Training samples: {len(training_data)}")
        print(f"Features: {len(feature_names)}")
        print(f"Num samples per explanation: {num_samples}")
    
    def explain_instance(
        self,
        x: np.ndarray,
        num_features: int = 10,
        top_labels: int = 1
    ) -> lime_tabular.LimeTabularExplainer:
        """
        Explain a single instance using LIME.
        
        Args:
            x: Input instance (1D array)
            num_features: Number of features to include in explanation
            top_labels: Number of top labels to explain
            
        Returns:
            LIME explanation object
        """
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten()
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            data_row=x,
            predict_fn=self.predict_fn,
            num_features=num_features,
            num_samples=self.num_samples,
            top_labels=top_labels
        )
        
        return explanation
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        num_features: int = 10
    ) -> pd.DataFrame:
        """
        Get aggregated feature importance across multiple instances.
        
        Args:
            X: Input data
            num_features: Number of top features to consider
            
        Returns:
            DataFrame with feature importance
        """
        feature_weights = {feature: [] for feature in self.feature_names}
        
        # Explain each instance
        for i in range(len(X)):
            explanation = self.explain_instance(X[i], num_features=len(self.feature_names))
            
            # Get feature weights for the positive class (disease)
            weights = dict(explanation.as_list(label=1))
            
            # Parse feature names and weights
            for feature_desc, weight in weights.items():
                # Extract feature name (LIME uses feature descriptions like "age <= 50")
                for feature in self.feature_names:
                    if feature in feature_desc:
                        feature_weights[feature].append(abs(weight))
                        break
        
        # Calculate mean importance
        importance_data = []
        for feature, weights in feature_weights.items():
            if weights:
                importance_data.append({
                    'feature': feature,
                    'importance': np.mean(weights),
                    'std': np.std(weights)
                })
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def explain_instance_detailed(
        self,
        x: np.ndarray,
        num_features: int = 10
    ) -> Dict:
        """
        Get detailed explanation for an instance.
        
        Args:
            x: Input instance
            num_features: Number of features to include
            
        Returns:
            Dictionary with detailed explanation
        """
        # Get LIME explanation
        explanation = self.explain_instance(x, num_features)
        
        # Get prediction
        pred_probs = self.predict_fn(x.reshape(1, -1))[0]
        
        # Get feature contributions for positive class
        feature_contributions = explanation.as_list(label=1)
        
        # Parse contributions
        contributions = []
        for feature_desc, weight in feature_contributions:
            contributions.append({
                'feature_description': feature_desc,
                'weight': weight,
                'contribution': 'positive' if weight > 0 else 'negative'
            })
        
        return {
            'prediction_proba': {
                self.class_names[0]: pred_probs[0],
                self.class_names[1]: pred_probs[1]
            },
            'predicted_class': self.class_names[1] if pred_probs[1] >= 0.5 else self.class_names[0],
            'contributions': contributions,
            'intercept': explanation.intercept[1] if hasattr(explanation, 'intercept') else None,
            'score': explanation.score if hasattr(explanation, 'score') else None
        }
    
    def save_explanation_html(
        self,
        x: np.ndarray,
        save_path: str,
        num_features: int = 10
    ) -> None:
        """
        Save LIME explanation as HTML.
        
        Args:
            x: Input instance
            save_path: Path to save HTML file
            num_features: Number of features to include
        """
        explanation = self.explain_instance(x, num_features)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as HTML
        explanation.save_to_file(str(save_path))
        
        print(f"LIME explanation saved to {save_path}")
    
    def compare_instances(
        self,
        X: np.ndarray,
        indices: List[int],
        num_features: int = 10
    ) -> pd.DataFrame:
        """
        Compare explanations for multiple instances.
        
        Args:
            X: Input data
            indices: Indices of instances to compare
            num_features: Number of features to include
            
        Returns:
            DataFrame with comparison
        """
        comparisons = []
        
        for idx in indices:
            explanation = self.explain_instance_detailed(X[idx], num_features)
            
            comparison = {
                'instance': idx,
                'predicted_class': explanation['predicted_class'],
                'disease_prob': explanation['prediction_proba'][self.class_names[1]]
            }
            
            # Add top contributing features
            for i, contrib in enumerate(explanation['contributions'][:5]):
                comparison[f'feature_{i+1}'] = contrib['feature_description']
                comparison[f'weight_{i+1}'] = contrib['weight']
            
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)


def load_model_and_data(config: Dict) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load trained model and data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, train_data, test_data, test_labels, feature_names)
    """
    # Load preprocessor
    preprocessor_path = Path(config['paths']['processed_data_dir']) / "preprocessor.pkl"
    preprocessor = DataPreprocessor.load(preprocessor_path)
    feature_names = preprocessor.feature_names
    
    # Load test data
    test_path = Path(config['paths']['processed_data_dir']) / "test_data.pt"
    test_data = torch.load(test_path)
    X_test = test_data['X'].numpy()
    y_test = test_data['y'].numpy()
    
    # Load training data (use first client's data as proxy)
    train_path = Path(config['paths']['partitions_dir']) / "client_0_train.pt"
    train_data = torch.load(train_path)
    X_train = train_data['X'].numpy()
    
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
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    return model, X_train, X_test, y_test, feature_names


def main():
    """Main function for LIME explanation generation."""
    parser = argparse.ArgumentParser(description="Generate LIME explanations")
    parser.add_argument("--config", type=str, default="configs/fl_config.yaml", help="Config file")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument("--num-instances", type=int, default=10, help="Number of instances to explain")
    parser.add_argument("--num-features", type=int, default=10, help="Number of features in explanation")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples for LIME")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and data
    print("Loading model and data...")
    model, X_train, X_test, y_test, feature_names = load_model_and_data(config)
    
    # Create LIME explainer
    print("\nCreating LIME explainer...")
    explainer = LIMEExplainer(
        model=model,
        training_data=X_train,
        feature_names=feature_names,
        num_samples=args.num_samples
    )
    
    # Explain instances
    print("\nGenerating explanations...")
    num_instances = min(args.num_instances, len(X_test))
    
    # Get feature importance across instances
    print("\nCalculating aggregated feature importance...")
    importance_df = explainer.get_feature_importance(X_test[:num_instances], args.num_features)
    print("\nTop 10 Most Important Features (LIME):")
    print(importance_df.head(10))
    
    # Save feature importance
    importance_path = Path(config['paths']['explanations_dir']) / "lime_feature_importance.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to {importance_path}")
    
    # Explain individual instances
    print("\nExplaining sample instances...")
    for i in range(min(5, num_instances)):
        explanation = explainer.explain_instance_detailed(X_test[i], args.num_features)
        
        print(f"\nInstance {i}:")
        print(f"  Predicted class: {explanation['predicted_class']}")
        print(f"  Disease probability: {explanation['prediction_proba']['Disease']:.4f}")
        print(f"  Top 5 contributing features:")
        for contrib in explanation['contributions'][:5]:
            print(f"    {contrib['feature_description']}: {contrib['weight']:.4f} ({contrib['contribution']})")
        
        # Save HTML explanation
        html_path = Path(config['paths']['explanations_dir']) / f"lime_instance_{i}.html"
        explainer.save_explanation_html(X_test[i], html_path, args.num_features)
    
    print("\nLIME explanation generation complete!")


if __name__ == "__main__":
    main()
