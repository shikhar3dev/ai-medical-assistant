"""Data Partitioning for Federated Learning."""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import yaml

from preprocessing.data_loader import load_heart_disease, load_diabetes
from preprocessing.preprocessor import DataPreprocessor, split_data


def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data in an IID (Independent and Identically Distributed) manner.
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        random_seed: Random seed
        
    Returns:
        List of (X, y) tuples for each client
    """
    np.random.seed(random_seed)
    
    # Shuffle indices
    indices = np.random.permutation(len(X))
    
    # Split indices into num_clients parts
    client_indices = np.array_split(indices, num_clients)
    
    # Create partitions
    partitions = []
    for idx in client_indices:
        partitions.append((X[idx], y[idx]))
    
    print(f"IID partitioning: {num_clients} clients")
    for i, (X_client, y_client) in enumerate(partitions):
        print(f"  Client {i}: {len(X_client)} samples")
    
    return partitions


def partition_heterogeneous(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data in a heterogeneous (non-IID) manner using Dirichlet distribution.
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        random_seed: Random seed
        
    Returns:
        List of (X, y) tuples for each client
    """
    np.random.seed(random_seed)
    
    # Get unique classes
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Initialize client data indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients using Dirichlet
    for c in classes:
        # Get indices of this class
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Calculate number of samples for each client
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        
        # Split indices according to proportions
        class_splits = np.split(class_indices, proportions)
        
        # Assign to clients
        for i, split in enumerate(class_splits):
            client_indices[i].extend(split)
    
    # Create partitions
    partitions = []
    for idx in client_indices:
        idx = np.array(idx)
        np.random.shuffle(idx)
        partitions.append((X[idx], y[idx]))
    
    print(f"Heterogeneous partitioning (α={alpha}): {num_clients} clients")
    for i, (X_client, y_client) in enumerate(partitions):
        class_dist = {int(c): int(np.sum(y_client == c)) for c in classes}
        print(f"  Client {i}: {len(X_client)} samples, class dist: {class_dist}")
    
    return partitions


def partition_pathological(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    shards_per_client: int = 2,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data in a pathological (extreme non-IID) manner.
    Each client gets only a few classes.
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        shards_per_client: Number of shards (class groups) per client
        random_seed: Random seed
        
    Returns:
        List of (X, y) tuples for each client
    """
    np.random.seed(random_seed)
    
    # Sort by label
    sorted_indices = np.argsort(y)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Create shards
    num_shards = num_clients * shards_per_client
    shard_size = len(X) // num_shards
    
    shard_indices = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else len(X)
        shard_indices.append(list(range(start, end)))
    
    # Shuffle shards
    np.random.shuffle(shard_indices)
    
    # Assign shards to clients
    client_indices = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(shards_per_client):
            shard_idx = i * shards_per_client + j
            if shard_idx < len(shard_indices):
                client_indices[i].extend(shard_indices[shard_idx])
    
    # Create partitions
    partitions = []
    for idx in client_indices:
        partitions.append((X_sorted[idx], y_sorted[idx]))
    
    print(f"Pathological partitioning ({shards_per_client} shards/client): {num_clients} clients")
    for i, (X_client, y_client) in enumerate(partitions):
        classes = np.unique(y_client)
        print(f"  Client {i}: {len(X_client)} samples, classes: {classes}")
    
    return partitions


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    strategy: str = 'heterogeneous',
    alpha: float = 0.5,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data according to specified strategy.
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        strategy: Partitioning strategy ('iid', 'heterogeneous', 'pathological')
        alpha: Dirichlet concentration parameter (for heterogeneous)
        random_seed: Random seed
        
    Returns:
        List of (X, y) tuples for each client
    """
    if strategy == 'iid':
        return partition_iid(X, y, num_clients, random_seed)
    elif strategy == 'heterogeneous':
        return partition_heterogeneous(X, y, num_clients, alpha, random_seed)
    elif strategy == 'pathological':
        return partition_pathological(X, y, num_clients, random_seed=random_seed)
    else:
        raise ValueError(f"Unknown partitioning strategy: {strategy}")


def save_partitions(
    partitions_train: List[Tuple[np.ndarray, np.ndarray]],
    partitions_val: List[Tuple[np.ndarray, np.ndarray]],
    save_dir: str
) -> None:
    """
    Save client partitions to disk.
    
    Args:
        partitions_train: List of training partitions
        partitions_val: List of validation partitions
        save_dir: Directory to save partitions
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for i, ((X_train, y_train), (X_val, y_val)) in enumerate(zip(partitions_train, partitions_val)):
        # Convert to tensors
        train_data = {
            'X': torch.FloatTensor(X_train),
            'y': torch.LongTensor(y_train)
        }
        val_data = {
            'X': torch.FloatTensor(X_val),
            'y': torch.LongTensor(y_val)
        }
        
        # Save
        torch.save(train_data, save_path / f"client_{i}_train.pt")
        torch.save(val_data, save_path / f"client_{i}_val.pt")
    
    print(f"Partitions saved to {save_path}")


def main():
    """Main function for data partitioning."""
    parser = argparse.ArgumentParser(description="Partition data for federated learning")
    parser.add_argument("--config", type=str, default="configs/fl_config.yaml", help="Config file")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--dataset", type=str, default="heart_disease", 
                       choices=["heart_disease", "diabetes"], help="Dataset to use")
    parser.add_argument("--strategy", type=str, default="heterogeneous",
                       choices=["iid", "heterogeneous", "pathological"], 
                       help="Partitioning strategy")
    parser.add_argument("--alpha", type=float, default=0.5, 
                       help="Dirichlet alpha for heterogeneous split")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.num_clients:
        num_clients = args.num_clients
    else:
        num_clients = config['federated']['min_clients']
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "heart_disease":
        X, y = load_heart_disease(config['paths']['raw_data_dir'])
    else:
        X, y = load_diabetes(config['paths']['raw_data_dir'])
    
    # Split into train/val/test
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['data']['random_seed']
    )
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor(
        scaling=config['data']['scaling'],
        handle_imbalance=config['data']['handle_imbalance'],
        imbalance_method=config['data']['imbalance_method'],
        random_seed=config['data']['random_seed']
    )
    
    # Fit on training data
    X_train_scaled, y_train_processed = preprocessor.fit_transform(
        X_train, y_train, apply_sampling=True
    )
    X_val_scaled, y_val_processed = preprocessor.transform(X_val, y_val)
    X_test_scaled, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # Save preprocessor
    preprocessor_path = Path(config['paths']['processed_data_dir']) / "preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    
    # Save test set
    test_data = {
        'X': torch.FloatTensor(X_test_scaled),
        'y': torch.LongTensor(y_test_processed)
    }
    test_path = Path(config['paths']['processed_data_dir']) / "test_data.pt"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_data, test_path)
    print(f"Test data saved to {test_path}")
    
    # Partition training data
    print(f"\nPartitioning training data ({args.strategy})...")
    partitions_train = partition_data(
        X_train_scaled,
        y_train_processed,
        num_clients=num_clients,
        strategy=args.strategy,
        alpha=args.alpha,
        random_seed=config['data']['random_seed']
    )
    
    # Partition validation data (same strategy)
    print(f"\nPartitioning validation data ({args.strategy})...")
    partitions_val = partition_data(
        X_val_scaled,
        y_val_processed,
        num_clients=num_clients,
        strategy=args.strategy,
        alpha=args.alpha,
        random_seed=config['data']['random_seed'] + 1  # Different seed for val
    )
    
    # Save partitions
    print("\nSaving partitions...")
    save_partitions(
        partitions_train,
        partitions_val,
        config['paths']['partitions_dir']
    )
    
    print("\nData partitioning complete!")
    print(f"  {num_clients} clients created")
    print(f"  Strategy: {args.strategy}")
    print(f"  Partitions saved to: {config['paths']['partitions_dir']}")


if __name__ == "__main__":
    main()
