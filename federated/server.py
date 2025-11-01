"""Federated Learning Server Implementation."""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import yaml

import torch
import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, Parameters
from flwr.server.strategy import FedAvg, FedProx, FedAdam
from flwr.server.client_manager import ClientManager

from federated.models import DiseasePredictor


class CustomFedAvg(FedAvg):
    """
    Custom FedAvg strategy with enhanced logging and model saving.
    
    Args:
        config: Configuration dictionary
        model: Initial model
        **kwargs: Additional FedAvg arguments
    """
    
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.model = model
        self.best_accuracy = 0.0
        self.best_auroc = 0.0
        self.round_num = 0
        
        # Create directories
        self.models_dir = Path(config['paths']['models_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients.
        
        Args:
            server_round: Current round number
            results: List of client results
            failures: List of failures
            
        Returns:
            Tuple of (aggregated parameters, metrics)
        """
        self.round_num = server_round
        
        # Log client metrics
        if results:
            print(f"\n{'='*60}")
            print(f"Round {server_round} - Training Results")
            print(f"{'='*60}")
            
            for client_proxy, fit_res in results:
                metrics = fit_res.metrics
                print(f"Client {metrics.get('client_id', 'Unknown')}: "
                      f"Loss={metrics.get('train_loss', 0):.4f}, "
                      f"Acc={metrics.get('train_accuracy', 0):.4f}")
        
        # Aggregate parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Save checkpoint
        if server_round % self.config['logging']['save_frequency'] == 0:
            self.save_checkpoint(aggregated_parameters, server_round)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: List of client results
            failures: List of failures
            
        Returns:
            Tuple of (aggregated loss, metrics)
        """
        if not results:
            return None, {}
        
        # Calculate weighted average metrics
        total_samples = sum([evaluate_res.num_examples for _, evaluate_res in results])
        
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Evaluation Results")
        print(f"{'='*60}")
        
        for client_proxy, evaluate_res in results:
            weight = evaluate_res.num_examples / total_samples
            weighted_loss += evaluate_res.loss * weight
            
            metrics = evaluate_res.metrics
            accuracy = metrics.get('val_accuracy', 0.0)
            weighted_accuracy += accuracy * weight
            
            print(f"Client {metrics.get('client_id', 'Unknown')}: "
                  f"Loss={evaluate_res.loss:.4f}, "
                  f"Acc={accuracy:.4f}")
        
        print(f"\nGlobal Metrics: Loss={weighted_loss:.4f}, Acc={weighted_accuracy:.4f}")
        print(f"{'='*60}\n")
        
        # Save best model
        if weighted_accuracy > self.best_accuracy:
            self.best_accuracy = weighted_accuracy
            print(f"New best accuracy: {self.best_accuracy:.4f}")
            self.save_best_model(server_round)
        
        aggregated_metrics = {
            "accuracy": weighted_accuracy,
            "loss": weighted_loss,
        }
        
        return weighted_loss, aggregated_metrics
    
    def save_checkpoint(self, parameters: Parameters, round_num: int) -> None:
        """
        Save model checkpoint.
        
        Args:
            parameters: Model parameters
            round_num: Current round number
        """
        # Convert parameters to state dict
        params_dict = zip(self.model.state_dict().keys(), parameters.tensors)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"
        torch.save({
            'round': round_num,
            'model_state_dict': state_dict,
            'best_accuracy': self.best_accuracy,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self, round_num: int) -> None:
        """
        Save the best model.
        
        Args:
            round_num: Current round number
        """
        best_model_path = self.models_dir / "best_model.pt"
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"
        
        if checkpoint_path.exists():
            import shutil
            shutil.copy(checkpoint_path, best_model_path)
            print(f"Best model saved: {best_model_path}")


def get_strategy(config: Dict, model: torch.nn.Module) -> fl.server.strategy.Strategy:
    """
    Get federated learning strategy.
    
    Args:
        config: Configuration dictionary
        model: Initial model
        
    Returns:
        Federated learning strategy
    """
    strategy_name = config['federated']['strategy']
    
    # Common strategy parameters
    strategy_params = {
        'fraction_fit': config['federated']['fraction_fit'],
        'fraction_evaluate': config['federated']['fraction_evaluate'],
        'min_fit_clients': config['federated']['min_fit_clients'],
        'min_evaluate_clients': config['federated']['min_evaluate_clients'],
        'min_available_clients': config['federated']['min_clients'],
    }
    
    # Get initial parameters
    initial_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)
    strategy_params['initial_parameters'] = initial_parameters
    
    if strategy_name == "FedAvg":
        return CustomFedAvg(config=config, model=model, **strategy_params)
    elif strategy_name == "FedProx":
        strategy_params['proximal_mu'] = 0.1
        return FedProx(**strategy_params)
    elif strategy_name == "FedAdam":
        return FedAdam(**strategy_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def start_server(config: Dict, model: torch.nn.Module) -> None:
    """
    Start the federated learning server.
    
    Args:
        config: Configuration dictionary
        model: Initial model
    """
    # Get strategy
    strategy = get_strategy(config, model)
    
    # Parse server address
    server_address = config['federated']['server_address']
    host, port = server_address.split(':')
    
    # Start server
    print(f"Starting Flower server on {server_address}")
    print(f"Waiting for {config['federated']['min_clients']} clients...")
    
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=config['federated']['num_rounds']),
        strategy=strategy,
    )


def main():
    """Main function to start the federated learning server."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--config", type=str, default="configs/fl_config.yaml", help="Config file path")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=None, help="Minimum number of clients")
    parser.add_argument("--server-address", type=str, default=None, help="Server address")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.rounds is not None:
        config['federated']['num_rounds'] = args.rounds
    if args.min_clients is not None:
        config['federated']['min_clients'] = args.min_clients
        config['federated']['min_fit_clients'] = args.min_clients
        config['federated']['min_evaluate_clients'] = args.min_clients
    if args.server_address is not None:
        config['federated']['server_address'] = args.server_address
    
    # Create initial model
    # Note: input_size will be determined by the first client
    # For now, use a placeholder value
    model = DiseasePredictor(
        input_size=13,  # Default for heart disease dataset
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout'],
        batch_norm=config['model']['batch_norm'],
        activation=config['model']['activation']
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Start server
    start_server(config, model)


if __name__ == "__main__":
    main()
