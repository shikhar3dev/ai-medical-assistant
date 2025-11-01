"""Data Loading Utilities for Medical Datasets."""

import argparse
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import requests
from io import StringIO
import yaml


def download_file(url: str, save_path: Path) -> None:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        save_path: Path to save the file
    """
    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {save_path}")


def load_heart_disease(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load UCI Heart Disease dataset.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    data_path = Path(data_dir) / "heart_disease.csv"
    
    if not data_path.exists():
        print(f"Heart disease dataset not found at {data_path}")
        print("Attempting to download...")
        download_heart_disease(data_dir)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    # Assuming last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Binarize target (0: no disease, 1: disease)
    y = (y > 0).astype(int)
    
    print(f"Loaded Heart Disease dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def download_heart_disease(data_dir: str = "data/raw") -> None:
    """
    Download UCI Heart Disease dataset.
    
    Args:
        data_dir: Directory to save the data
    """
    # UCI Heart Disease dataset
    # Using Cleveland dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    save_path = Path(data_dir) / "heart_disease.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Column names
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        # Download data
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse data
        data = response.text
        df = pd.read_csv(StringIO(data), names=columns, na_values='?')
        
        # Handle missing values
        df = df.dropna()
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Heart Disease dataset downloaded and saved to {save_path}")
        
    except Exception as e:
        print(f"Error downloading Heart Disease dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        create_synthetic_heart_disease(save_path)


def create_synthetic_heart_disease(save_path: Path, n_samples: int = 300) -> None:
    """
    Create synthetic heart disease dataset.
    
    Args:
        save_path: Path to save the dataset
        n_samples: Number of samples to generate
    """
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
    }
    
    # Generate target with some correlation to features
    risk_score = (
        (data['age'] > 55).astype(int) +
        (data['cp'] > 1).astype(int) +
        (data['trestbps'] > 140).astype(int) +
        (data['chol'] > 240).astype(int) +
        (data['thalach'] < 120).astype(int) +
        data['exang']
    )
    
    data['target'] = (risk_score > 2).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Synthetic Heart Disease dataset created: {save_path}")


def load_diabetes(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Pima Indians Diabetes dataset.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    data_path = Path(data_dir) / "diabetes.csv"
    
    if not data_path.exists():
        print(f"Diabetes dataset not found at {data_path}")
        print("Attempting to download...")
        download_diabetes(data_dir)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"Loaded Diabetes dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def download_diabetes(data_dir: str = "data/raw") -> None:
    """
    Download Pima Indians Diabetes dataset.
    
    Args:
        data_dir: Directory to save the data
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    save_path = Path(data_dir) / "diabetes.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
    ]
    
    try:
        # Download data
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse data
        data = response.text
        df = pd.read_csv(StringIO(data), names=columns)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Diabetes dataset downloaded and saved to {save_path}")
        
    except Exception as e:
        print(f"Error downloading Diabetes dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        create_synthetic_diabetes(save_path)


def create_synthetic_diabetes(save_path: Path, n_samples: int = 768) -> None:
    """
    Create synthetic diabetes dataset.
    
    Args:
        save_path: Path to save the dataset
        n_samples: Number of samples to generate
    """
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'pregnancies': np.random.randint(0, 17, n_samples),
        'glucose': np.random.randint(0, 200, n_samples),
        'blood_pressure': np.random.randint(0, 122, n_samples),
        'skin_thickness': np.random.randint(0, 99, n_samples),
        'insulin': np.random.randint(0, 846, n_samples),
        'bmi': np.random.uniform(0, 67, n_samples),
        'diabetes_pedigree': np.random.uniform(0.078, 2.42, n_samples),
        'age': np.random.randint(21, 81, n_samples),
    }
    
    # Generate target with some correlation
    risk_score = (
        (data['glucose'] > 140).astype(int) +
        (data['bmi'] > 30).astype(int) +
        (data['age'] > 45).astype(int) +
        (data['diabetes_pedigree'] > 0.5).astype(int)
    )
    
    data['outcome'] = (risk_score > 1).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Synthetic Diabetes dataset created: {save_path}")


def download_datasets(data_dir: str = "data/raw") -> None:
    """
    Download all datasets.
    
    Args:
        data_dir: Directory to save the data
    """
    print("Downloading datasets...")
    download_heart_disease(data_dir)
    download_diabetes(data_dir)
    print("All datasets downloaded!")


def main():
    """Main function for data loading."""
    parser = argparse.ArgumentParser(description="Download medical datasets")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--dataset", type=str, choices=["heart", "diabetes", "all"], 
                       default="all", help="Dataset to download")
    args = parser.parse_args()
    
    if args.download:
        if args.dataset == "heart":
            download_heart_disease(args.data_dir)
        elif args.dataset == "diabetes":
            download_diabetes(args.data_dir)
        else:
            download_datasets(args.data_dir)
    else:
        # Test loading
        print("\nTesting Heart Disease dataset:")
        X_heart, y_heart = load_heart_disease(args.data_dir)
        print(f"Shape: {X_heart.shape}")
        
        print("\nTesting Diabetes dataset:")
        X_diabetes, y_diabetes = load_diabetes(args.data_dir)
        print(f"Shape: {X_diabetes.shape}")


if __name__ == "__main__":
    main()
