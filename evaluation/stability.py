"""Explanation Stability Tracking"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class ExplanationStabilityTracker:
    """
    Track stability of explanations across federated learning rounds.
    
    Args:
        feature_names: List of feature names
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.importance_history = []
        self.round_numbers = []
        
    def add_round(self, round_num: int, importance_df: pd.DataFrame) -> None:
        """
        Add feature importance from a round.
        
        Args:
            round_num: Round number
            importance_df: Feature importance DataFrame
        """
        self.round_numbers.append(round_num)
        self.importance_history.append(importance_df.copy())
    
    def calculate_stability_score(self, top_k: int = 10) -> float:
        """
        Calculate stability score based on top-k feature consistency.
        
        Args:
            top_k: Number of top features to consider
            
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(self.importance_history) < 2:
            return 1.0
        
        # Get top-k features from each round
        top_features_per_round = []
        for importance_df in self.importance_history:
            top_features = set(importance_df.head(top_k)['feature'].tolist())
            top_features_per_round.append(top_features)
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(top_features_per_round) - 1):
            set1 = top_features_per_round[i]
            set2 = top_features_per_round[i + 1]
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            jaccard = intersection / union if union > 0 else 0
            similarities.append(jaccard)
        
        # Return average similarity
        return np.mean(similarities)
    
    def calculate_rank_correlation(self, top_k: int = 10) -> List[float]:
        """
        Calculate rank correlation between consecutive rounds.
        
        Args:
            top_k: Number of top features to consider
            
        Returns:
            List of Spearman correlation coefficients
        """
        from scipy.stats import spearmanr
        
        if len(self.importance_history) < 2:
            return []
        
        correlations = []
        
        for i in range(len(self.importance_history) - 1):
            # Get importance values for common features
            df1 = self.importance_history[i].head(top_k).set_index('feature')['importance']
            df2 = self.importance_history[i + 1].head(top_k).set_index('feature')['importance']
            
            # Find common features
            common_features = df1.index.intersection(df2.index)
            
            if len(common_features) > 1:
                corr, _ = spearmanr(df1[common_features], df2[common_features])
                correlations.append(corr)
        
        return correlations
    
    def get_feature_rank_history(self, feature: str) -> List[int]:
        """
        Get rank history for a specific feature.
        
        Args:
            feature: Feature name
            
        Returns:
            List of ranks across rounds
        """
        ranks = []
        
        for importance_df in self.importance_history:
            feature_row = importance_df[importance_df['feature'] == feature]
            if not feature_row.empty:
                ranks.append(feature_row['rank'].values[0])
            else:
                ranks.append(len(self.feature_names))  # Worst rank if not found
        
        return ranks
    
    def plot_stability(self, top_k: int = 10, save_path: Optional[str] = None) -> None:
        """
        Plot stability metrics over rounds.
        
        Args:
            top_k: Number of top features to track
            save_path: Path to save the plot
        """
        if len(self.importance_history) < 2:
            print("Not enough rounds to plot stability")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Feature importance over rounds
        final_importance = self.importance_history[-1]
        top_features = final_importance.head(top_k)['feature'].tolist()
        
        for feature in top_features:
            importances = []
            for importance_df in self.importance_history:
                feature_row = importance_df[importance_df['feature'] == feature]
                if not feature_row.empty:
                    importances.append(feature_row['importance'].values[0])
                else:
                    importances.append(0)
            
            axes[0].plot(self.round_numbers, importances, marker='o', label=feature, alpha=0.7)
        
        axes[0].set_xlabel('Federated Learning Round')
        axes[0].set_ylabel('Feature Importance')
        axes[0].set_title(f'Top {top_k} Feature Importance Across Rounds')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Stability score over rounds
        stability_scores = []
        for i in range(1, len(self.importance_history)):
            # Calculate stability between round i-1 and i
            set1 = set(self.importance_history[i-1].head(top_k)['feature'].tolist())
            set2 = set(self.importance_history[i].head(top_k)['feature'].tolist())
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0
            
            stability_scores.append(jaccard)
        
        axes[1].plot(self.round_numbers[1:], stability_scores, marker='o', linewidth=2, color='steelblue')
        axes[1].axhline(y=0.8, color='green', linestyle='--', label='High Stability (0.8)')
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='Medium Stability (0.5)')
        axes[1].set_xlabel('Federated Learning Round')
        axes[1].set_ylabel('Jaccard Similarity')
        axes[1].set_title(f'Explanation Stability (Top {top_k} Features)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stability plot saved to {save_path}")
        
        plt.show()
    
    def get_stability_report(self, top_k: int = 10) -> Dict:
        """
        Generate stability report.
        
        Args:
            top_k: Number of top features to consider
            
        Returns:
            Dictionary with stability metrics
        """
        stability_score = self.calculate_stability_score(top_k)
        rank_correlations = self.calculate_rank_correlation(top_k)
        
        report = {
            'num_rounds': len(self.importance_history),
            'stability_score': stability_score,
            'mean_rank_correlation': np.mean(rank_correlations) if rank_correlations else None,
            'min_rank_correlation': np.min(rank_correlations) if rank_correlations else None,
            'max_rank_correlation': np.max(rank_correlations) if rank_correlations else None,
        }
        
        return report
    
    def save(self, save_path: str) -> None:
        """
        Save tracker to file.
        
        Args:
            save_path: Path to save the tracker
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Stability tracker saved to {save_path}")
    
    @staticmethod
    def load(load_path: str) -> 'ExplanationStabilityTracker':
        """
        Load tracker from file.
        
        Args:
            load_path: Path to load the tracker from
            
        Returns:
            Loaded tracker
        """
        with open(load_path, 'rb') as f:
            tracker = pickle.load(f)
        
        print(f"Stability tracker loaded from {load_path}")
        return tracker


def compare_explanation_methods(
    shap_importance_history: List[pd.DataFrame],
    lime_importance_history: List[pd.DataFrame],
    feature_names: List[str],
    top_k: int = 10,
    save_path: Optional[str] = None
) -> Dict:
    """
    Compare stability of SHAP and LIME explanations.
    
    Args:
        shap_importance_history: List of SHAP importance DataFrames
        lime_importance_history: List of LIME importance DataFrames
        feature_names: List of feature names
        top_k: Number of top features to consider
        save_path: Path to save comparison plot
        
    Returns:
        Dictionary with comparison metrics
    """
    # Create trackers
    shap_tracker = ExplanationStabilityTracker(feature_names)
    lime_tracker = ExplanationStabilityTracker(feature_names)
    
    # Add rounds
    for i, (shap_imp, lime_imp) in enumerate(zip(shap_importance_history, lime_importance_history)):
        shap_tracker.add_round(i, shap_imp)
        lime_tracker.add_round(i, lime_imp)
    
    # Get stability scores
    shap_stability = shap_tracker.calculate_stability_score(top_k)
    lime_stability = lime_tracker.calculate_stability_score(top_k)
    
    # Get rank correlations
    shap_corr = shap_tracker.calculate_rank_correlation(top_k)
    lime_corr = lime_tracker.calculate_rank_correlation(top_k)
    
    # Create comparison
    comparison = {
        'shap_stability': shap_stability,
        'lime_stability': lime_stability,
        'shap_mean_correlation': np.mean(shap_corr) if shap_corr else None,
        'lime_mean_correlation': np.mean(lime_corr) if lime_corr else None,
    }
    
    # Plot comparison
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['SHAP', 'LIME']
        stability_scores = [shap_stability, lime_stability]
        
        ax.bar(methods, stability_scores, alpha=0.7, color=['steelblue', 'coral'])
        ax.set_ylabel('Stability Score')
        ax.set_title(f'Explanation Stability Comparison (Top {top_k} Features)')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(stability_scores):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()
    
    return comparison
