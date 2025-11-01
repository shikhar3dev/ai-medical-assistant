"""Visualization Tools for Explainability."""

from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 10,
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
) -> None:
    """
    Plot feature importance as a bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_k: Number of top features to plot
        save_path: Path to save the plot
        title: Plot title
    """
    # Get top K features
    top_features = importance_df.head(top_k)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    plot_type: str = 'dot'
) -> None:
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature values
        feature_names: List of feature names
        save_path: Path to save the plot
        plot_type: Type of plot ('dot', 'bar', 'violin')
    """
    plt.figure(figsize=(10, 8))
    
    if plot_type == 'dot':
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    elif plot_type == 'bar':
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', show=False)
    elif plot_type == 'violin':
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='violin', show=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.show()


def plot_shap_waterfall(
    shap_values: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_display: int = 10
) -> None:
    """
    Plot SHAP waterfall plot for a single instance.
    
    Args:
        shap_values: SHAP values for the instance
        base_value: Base value (expected value)
        feature_values: Feature values for the instance
        feature_names: List of feature names
        save_path: Path to save the plot
        max_display: Maximum number of features to display
    """
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to {save_path}")
    
    plt.show()


def plot_lime_explanation(
    feature_weights: List[tuple],
    save_path: Optional[str] = None,
    title: str = "LIME Feature Contributions"
) -> None:
    """
    Plot LIME explanation as a bar chart.
    
    Args:
        feature_weights: List of (feature_name, weight) tuples
        save_path: Path to save the plot
        title: Plot title
    """
    # Sort by absolute weight
    feature_weights = sorted(feature_weights, key=lambda x: abs(x[1]))
    
    features = [f[0] for f in feature_weights]
    weights = [f[1] for f in feature_weights]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    colors = ['green' if w > 0 else 'red' for w in weights]
    plt.barh(range(len(features)), weights, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Weight')
    plt.ylabel('Feature')
    plt.title(title)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LIME plot saved to {save_path}")
    
    plt.show()


def plot_explanation_comparison(
    shap_importance: pd.DataFrame,
    lime_importance: pd.DataFrame,
    top_k: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Compare SHAP and LIME feature importance.
    
    Args:
        shap_importance: SHAP feature importance DataFrame
        lime_importance: LIME feature importance DataFrame
        top_k: Number of top features to compare
        save_path: Path to save the plot
    """
    # Get top K features from each
    shap_top = shap_importance.head(top_k).set_index('feature')['importance']
    lime_top = lime_importance.head(top_k).set_index('feature')['importance']
    
    # Combine
    comparison = pd.DataFrame({
        'SHAP': shap_top,
        'LIME': lime_top
    }).fillna(0)
    
    # Normalize for comparison
    comparison['SHAP'] = comparison['SHAP'] / comparison['SHAP'].max()
    comparison['LIME'] = comparison['LIME'] / comparison['LIME'].max()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison.plot(kind='barh', ax=ax, alpha=0.8)
    plt.xlabel('Normalized Importance')
    plt.ylabel('Feature')
    plt.title('SHAP vs LIME Feature Importance Comparison')
    plt.legend(loc='best')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def plot_explanation_stability(
    importance_history: List[pd.DataFrame],
    top_k: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot stability of feature importance across FL rounds.
    
    Args:
        importance_history: List of importance DataFrames from different rounds
        top_k: Number of top features to track
        save_path: Path to save the plot
    """
    # Get top features from the last round
    final_importance = importance_history[-1]
    top_features = final_importance.head(top_k)['feature'].tolist()
    
    # Track importance over rounds
    importance_over_time = {feature: [] for feature in top_features}
    
    for round_importance in importance_history:
        for feature in top_features:
            feature_row = round_importance[round_importance['feature'] == feature]
            if not feature_row.empty:
                importance_over_time[feature].append(feature_row['importance'].values[0])
            else:
                importance_over_time[feature].append(0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    rounds = list(range(1, len(importance_history) + 1))
    
    for feature, importances in importance_over_time.items():
        plt.plot(rounds, importances, marker='o', label=feature, alpha=0.7)
    
    plt.xlabel('Federated Learning Round')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Stability Across FL Rounds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stability plot saved to {save_path}")
    
    plt.show()


def plot_prediction_distribution(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of predictions.
    
    Args:
        predictions: Predicted probabilities
        true_labels: True labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Separate by true label
    pred_negative = predictions[true_labels == 0]
    pred_positive = predictions[true_labels == 1]
    
    plt.hist(pred_negative, bins=30, alpha=0.5, label='True Negative', color='blue')
    plt.hist(pred_positive, bins=30, alpha=0.5, label='True Positive', color='red')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()


def plot_feature_correlation_heatmap(
    X: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    Plot correlation heatmap of features.
    
    Args:
        X: Feature DataFrame
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()


def plot_client_data_distribution(
    client_data: Dict[int, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot data distribution across clients.
    
    Args:
        client_data: Dictionary mapping client_id to labels
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sample counts per client
    client_ids = list(client_data.keys())
    sample_counts = [len(labels) for labels in client_data.values()]
    
    axes[0].bar(client_ids, sample_counts, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Sample Distribution Across Clients')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Class distribution per client
    width = 0.35
    x = np.arange(len(client_ids))
    
    class_0_counts = [np.sum(labels == 0) for labels in client_data.values()]
    class_1_counts = [np.sum(labels == 1) for labels in client_data.values()]
    
    axes[1].bar(x - width/2, class_0_counts, width, label='Class 0', alpha=0.7, color='blue')
    axes[1].bar(x + width/2, class_1_counts, width, label='Class 1', alpha=0.7, color='red')
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Class Distribution Across Clients')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(client_ids)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Client distribution plot saved to {save_path}")
    
    plt.show()


def create_explanation_report(
    shap_importance: pd.DataFrame,
    lime_importance: pd.DataFrame,
    save_dir: str
) -> None:
    """
    Create a comprehensive explanation report with multiple visualizations.
    
    Args:
        shap_importance: SHAP feature importance
        lime_importance: LIME feature importance
        save_dir: Directory to save plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating explanation report...")
    
    # 1. SHAP feature importance
    plot_feature_importance(
        shap_importance,
        top_k=15,
        save_path=save_path / "shap_feature_importance.png",
        title="SHAP Feature Importance"
    )
    
    # 2. LIME feature importance
    plot_feature_importance(
        lime_importance,
        top_k=15,
        save_path=save_path / "lime_feature_importance.png",
        title="LIME Feature Importance"
    )
    
    # 3. Comparison
    plot_explanation_comparison(
        shap_importance,
        lime_importance,
        top_k=15,
        save_path=save_path / "shap_lime_comparison.png"
    )
    
    print(f"Explanation report saved to {save_dir}")
