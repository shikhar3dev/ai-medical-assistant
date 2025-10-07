"""Federated Learning Client Implementation"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar

from federated.models import DiseasePredictor, get_loss_function, get_optimizer, get_lr_scheduler
from privacy.differential_privacy import DPTrainer


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning.
    
    Args:
        client_id: Unique identifier for this client
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        device: Device to run training on
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training configuration
        self.local_epochs = config['training']['local_epochs']
        self.learning_rate = config['training']['learning_rate']
        
        # Loss function
        self.criterion = get_loss_function(
            config['training']['loss_function']
        )
        
        # Optimizer
        self.optimizer = get_optimizer(
            self.model,
            config['training']['optimizer'],
            lr=self.learning_rate,
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config['training']['lr_scheduler'],
            step_size=config['training'].get('lr_step_size', 10),
            gamma=config['training'].get('lr_decay', 0.95)
        )
        
        # Privacy
        self.use_dp = config['privacy']['enable_dp']
        if self.use_dp:
            self.dp_trainer = DPTrainer(
                model=self.model,
                optimizer=self.optimizer,
                max_grad_norm=config['privacy']['max_grad_norm'],
                noise_multiplier=config['privacy'].get('noise_multiplier', 1.1),
                epsilon=config['privacy']['epsilon'],
                delta=config['privacy']['delta']
            )
        
        print(f"[Client {self.client_id}] Initialized with {len(train_loader.dataset)} training samples")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Get model parameters as NumPy arrays.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of model parameters as NumPy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from NumPy arrays.
        
        Args:
            parameters: List of model parameters as NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (updated parameters, number of samples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train the model
        train_loss, train_acc = self.train()
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config={})
        
        # Return results
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "client_id": self.client_id
        }
        
        return updated_parameters, len(self.train_loader.dataset), metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration from server
            
        Returns:
            Tuple of (loss, number of samples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        val_loss, val_acc = self.validate()
        
        # Return results
        metrics = {
            "val_accuracy": float(val_acc),
            "client_id": self.client_id
        }
        
        return float(val_loss), len(self.val_loader.dataset), metrics
    
    def train(self) -> Tuple[float, float]:
        """
        Train the model for one federated round.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                if self.use_dp:
                    # Use differential privacy
                    self.dp_trainer.step(loss)
                else:
                    # Standard training
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                
                # Calculate accuracy
                pred = torch.sigmoid(output) >= 0.5
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            total_loss += epoch_loss / len(self.train_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        avg_loss = total_loss / self.local_epochs
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"[Client {self.client_id}] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model on local validation data.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Track metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = torch.sigmoid(output) >= 0.5
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"[Client {self.client_id}] Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy


def load_client_data(client_id: int, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Load data for a specific client.
    
    Args:
        client_id: Client identifier
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    partitions_dir = Path(config['paths']['partitions_dir'])
    
    # Load client data
    train_data = torch.load(partitions_dir / f"client_{client_id}_train.pt")
    val_data = torch.load(partitions_dir / f"client_{client_id}_val.pt")
    
    # Create datasets
    train_dataset = TensorDataset(train_data['X'], train_data['y'])
    val_dataset = TensorDataset(val_data['X'], val_data['y'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    return train_loader, val_loader


def main():
    """Main function to start a federated learning client."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    parser.add_argument("--config", type=str, default="configs/fl_config.yaml", help="Config file path")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override server address if provided
    if args.server_address:
        config['federated']['server_address'] = args.server_address
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client {args.client_id}] Using device: {device}")
    
    # Load client data
    train_loader, val_loader = load_client_data(args.client_id, config)
    
    # Get input size from data
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]
    config['model']['input_size'] = input_size
    
    # Create model
    model = DiseasePredictor(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout'],
        batch_norm=config['model']['batch_norm'],
        activation=config['model']['activation']
    )
    
    # Create Flower client
    client = FlowerClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Start client
    print(f"[Client {args.client_id}] Connecting to server at {config['federated']['server_address']}")
    fl.client.start_numpy_client(
        server_address=config['federated']['server_address'],
        client=client
    )


if __name__ == "__main__":
    main()
