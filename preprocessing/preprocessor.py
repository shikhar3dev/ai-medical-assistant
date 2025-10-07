"""Data Preprocessing and Feature Engineering"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import pickle
from pathlib import Path


class DataPreprocessor:
    """
    Data preprocessor for medical datasets.
    
    Args:
        scaling: Scaling method ('standard', 'minmax', 'robust', 'none')
        handle_imbalance: Whether to handle class imbalance
        imbalance_method: Method for handling imbalance ('smote', 'oversample', 'undersample')
        random_seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        scaling: str = 'standard',
        handle_imbalance: bool = True,
        imbalance_method: str = 'smote',
        random_seed: int = 42
    ):
        self.scaling = scaling
        self.handle_imbalance = handle_imbalance
        self.imbalance_method = imbalance_method
        self.random_seed = random_seed
        
        # Initialize scaler
        if scaling == 'standard':
            self.scaler = StandardScaler()
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'robust':
            self.scaler = RobustScaler()
        elif scaling == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {scaling}")
        
        # Initialize imbalance handler
        if handle_imbalance:
            if imbalance_method == 'smote':
                self.sampler = SMOTE(random_state=random_seed)
            elif imbalance_method == 'oversample':
                from imblearn.over_sampling import RandomOverSampler
                self.sampler = RandomOverSampler(random_state=random_seed)
            elif imbalance_method == 'undersample':
                self.sampler = RandomUnderSampler(random_state=random_seed)
            elif imbalance_method == 'smote_tomek':
                self.sampler = SMOTETomek(random_state=random_seed)
            else:
                raise ValueError(f"Unknown imbalance method: {imbalance_method}")
        else:
            self.sampler = None
        
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(X)
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        apply_sampling: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            apply_sampling: Whether to apply imbalance sampling
            
        Returns:
            Tuple of (transformed X, transformed y)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Handle class imbalance (only for training data)
        if apply_sampling and self.sampler is not None and y is not None:
            X_scaled, y = self.sampler.fit_resample(X_scaled, y)
            y = y.values if hasattr(y, 'values') else y
        elif y is not None:
            y = y.values if hasattr(y, 'values') else y
        
        return X_scaled, y
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        apply_sampling: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            apply_sampling: Whether to apply imbalance sampling
            
        Returns:
            Tuple of (transformed X, transformed y)
        """
        self.fit(X, y)
        return self.transform(X, y, apply_sampling)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        X = X.copy()
        
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        return X
    
    def save(self, path: str) -> None:
        """
        Save preprocessor to file.
        
        Args:
            path: Path to save the preprocessor
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Preprocessor saved to {save_path}")
    
    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        """
        Load preprocessor from file.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        print(f"Preprocessor loaded from {path}")
        return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        test_ratio: Proportion of test data
        random_seed: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=y
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=y_temp
    )
    
    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({test_ratio*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_info(X: pd.DataFrame) -> pd.DataFrame:
    """
    Get information about features.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        DataFrame with feature information
    """
    info = pd.DataFrame({
        'feature': X.columns,
        'dtype': X.dtypes.values,
        'missing': X.isnull().sum().values,
        'missing_pct': (X.isnull().sum() / len(X) * 100).values,
        'unique': X.nunique().values,
        'mean': X.select_dtypes(include=[np.number]).mean().reindex(X.columns).values,
        'std': X.select_dtypes(include=[np.number]).std().reindex(X.columns).values,
        'min': X.select_dtypes(include=[np.number]).min().reindex(X.columns).values,
        'max': X.select_dtypes(include=[np.number]).max().reindex(X.columns).values,
    })
    
    return info


def detect_outliers(
    X: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in the data.
    
    Args:
        X: Feature DataFrame
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean DataFrame indicating outliers
    """
    outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = (X[col] < lower_bound) | (X[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
            outliers[col] = z_scores > threshold
    
    return outliers


def remove_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'iqr',
    threshold: float = 3.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers from the data.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Outlier detection method
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (X without outliers, y without outliers)
    """
    outliers = detect_outliers(X, method, threshold)
    
    # Remove rows with any outliers
    mask = ~outliers.any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    
    n_removed = len(X) - len(X_clean)
    print(f"Removed {n_removed} outliers ({n_removed/len(X)*100:.2f}%)")
    
    return X_clean, y_clean
