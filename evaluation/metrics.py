"""Model Evaluation Metrics."""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_pred_proba),
        'auprc': average_precision_score(y_true, y_pred_proba),
    }
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on given data.
    
    Args:
        model: Neural network model
        X: Input features
        y: True labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        outputs = model(X)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs >= threshold).astype(int)
        
        # Convert labels
        y_true = y.numpy().flatten()
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, preds, probs)
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ['No Disease', 'Disease'],
    save_path: Optional[str] = None,
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {auprc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save the plot
    """
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    
    plt.show()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ['No Disease', 'Disease'],
    save_path: Optional[str] = None
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
        
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")
    
    return report


def create_evaluation_report(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    save_dir: str,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Create comprehensive evaluation report with plots and metrics.
    
    Args:
        model: Neural network model
        X: Input features
        y: True labels
        save_dir: Directory to save results
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating evaluation report...")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs >= threshold).astype(int)
    
    y_true = y.numpy().flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, preds, probs)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    print("=" * 50)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_path / "metrics.csv", index=False)
    
    # Generate plots
    plot_confusion_matrix(y_true, preds, save_path=save_path / "confusion_matrix.png")
    plot_confusion_matrix(y_true, preds, save_path=save_path / "confusion_matrix_normalized.png", normalize=True)
    plot_roc_curve(y_true, probs, save_path=save_path / "roc_curve.png")
    plot_precision_recall_curve(y_true, probs, save_path=save_path / "pr_curve.png")
    plot_calibration_curve(y_true, probs, save_path=save_path / "calibration_curve.png")
    
    # Generate classification report
    report = generate_classification_report(y_true, preds, save_path=save_path / "classification_report.txt")
    print("\nClassification Report:")
    print(report)
    
    print(f"\nEvaluation report saved to {save_dir}")
    
    return metrics


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        save_path: Path to save comparison
        
    Returns:
        Comparison DataFrame
    """
    comparison_df = pd.DataFrame(results).T
    
    if save_path:
        comparison_df.to_csv(save_path)
        print(f"Model comparison saved to {save_path}")
    
    return comparison_df


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with 'loss' and 'accuracy' lists
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()
