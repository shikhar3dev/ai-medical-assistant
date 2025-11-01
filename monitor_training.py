"""Simple training monitor."""

import time
import os
from pathlib import Path

def monitor_training():
    print("🔄 Monitoring Federated Learning Training...")
    print("=" * 50)
    
    models_dir = Path("models")
    checkpoints_dir = Path("checkpoints")
    
    round_count = 0
    start_time = time.time()
    
    while True:
        # Check for checkpoints
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("checkpoint_round_*.pt"))
            if len(checkpoints) > round_count:
                round_count = len(checkpoints)
                elapsed = time.time() - start_time
                print(f"✅ Round {round_count} completed! ({elapsed:.1f}s elapsed)")
        
        # Check for final model
        if models_dir.exists():
            model_files = list(models_dir.glob("best_model.pt"))
            if model_files:
                elapsed = time.time() - start_time
                print(f"\n🎉 TRAINING COMPLETE! ({elapsed:.1f}s total)")
                print(f"✅ Model saved: {model_files[0]}")
                print("\n📊 Next steps:")
                print("1. Refresh dashboard (F5 in browser)")
                print("2. View model performance and explanations")
                print("3. Make predictions on new patients")
                break
        
        # Show progress
        elapsed = time.time() - start_time
        if elapsed > 0 and elapsed % 30 == 0:  # Every 30 seconds
            print(f"⏳ Training in progress... ({elapsed:.0f}s elapsed)")
        
        time.sleep(5)  # Check every 5 seconds
        
        # Timeout after 15 minutes
        if elapsed > 900:
            print("⚠️ Training taking longer than expected. Check terminal windows.")
            break

if __name__ == "__main__":
    monitor_training()
