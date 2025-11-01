"""End-to-End Experiment Runner for Federated Learning"""

import argparse
import subprocess
import time
import yaml
from pathlib import Path
import sys

def run_command(cmd, cwd=None, wait=True):
    """
    Run a command.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        wait: Whether to wait for completion
        
    Returns:
        Process object
    """
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    
    if wait:
        result = subprocess.run(cmd, shell=True, cwd=cwd)
        return result.returncode
    else:
        process = subprocess.Popen(cmd, shell=True, cwd=cwd)
        return process


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run FL experiment end-to-end")
    parser.add_argument("--config", type=str, default="configs/fl_config.yaml", help="Config file")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--dataset", type=str, default="heart_disease", 
                       choices=["heart_disease", "diabetes"], help="Dataset to use")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-training", action="store_true", help="Skip FL training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("FEDERATED LEARNING EXPERIMENT")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.rounds}")
    print("="*60 + "\n")
    
    # Step 1: Download and prepare data
    if not args.skip_data:
        print("\n### STEP 1: DATA PREPARATION ###\n")
        
        # Download datasets
        print("Downloading datasets...")
        ret = run_command(
            f"python preprocessing/data_loader.py --download --dataset {args.dataset}"
        )
        if ret != 0:
            print("Warning: Data download may have failed. Continuing with synthetic data...")
        
        # Partition data
        print("\nPartitioning data for federated learning...")
        ret = run_command(
            f"python preprocessing/partitioner.py --num-clients {args.num_clients} "
            f"--dataset {args.dataset} --strategy heterogeneous"
        )
        if ret != 0:
            print("Error: Data partitioning failed!")
            return 1
        
        print("\n✅ Data preparation complete!")
    
    # Step 2: Run federated training
    if not args.skip_training:
        print("\n### STEP 2: FEDERATED TRAINING ###\n")
        print("Note: This will start the FL server and clients.")
        print("In a real deployment, run server and clients in separate terminals.\n")
        
        print("For manual training, run these commands in separate terminals:")
        print("\nTerminal 1 (Server):")
        print(f"  python federated/server.py --rounds {args.rounds} --min-clients {args.num_clients}")
        
        for i in range(args.num_clients):
            print(f"\nTerminal {i+2} (Client {i}):")
            print(f"  python federated/client.py --client-id {i}")
        
        print("\n" + "="*60)
        print("Automated training is not supported in this script.")
        print("Please run the server and clients manually as shown above.")
        print("="*60 + "\n")
    
    # Step 3: Generate explanations
    if not args.skip_evaluation:
        print("\n### STEP 3: GENERATING EXPLANATIONS ###\n")
        
        # Check if model exists
        model_path = Path(config['paths']['models_dir']) / "best_model.pt"
        if not model_path.exists():
            print(f"⚠️  Model not found at {model_path}")
            print("Please train the model first before generating explanations.")
            return 1
        
        # Generate SHAP explanations
        print("Generating SHAP explanations...")
        ret = run_command(
            "python explainability/shap_explainer.py --num-samples 100"
        )
        if ret != 0:
            print("Warning: SHAP explanation generation failed!")
        
        # Generate LIME explanations
        print("\nGenerating LIME explanations...")
        ret = run_command(
            "python explainability/lime_explainer.py --num-instances 10"
        )
        if ret != 0:
            print("Warning: LIME explanation generation failed!")
        
        print("\n✅ Explanation generation complete!")
    
    # Step 4: Launch dashboard
    print("\n### STEP 4: LAUNCHING DASHBOARD ###\n")
    print("To view results in the interactive dashboard, run:")
    print("\n  streamlit run dashboard/app.py\n")
    print("="*60 + "\n")
    
    print("✅ Experiment setup complete!")
    print("\nNext steps:")
    print("1. Train the model using FL server and clients (if not done)")
    print("2. Generate explanations (if not done)")
    print("3. Launch the dashboard to visualize results")
    print("\nFor detailed instructions, see README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
