"""Check training status and show progress."""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("FEDERATED LEARNING TRAINING STATUS")
print("="*60)
print()

# Check if processes are running
print("📊 Checking training progress...")
print()

# Check for model files
models_dir = Path("models")
checkpoints_dir = Path("checkpoints")

if models_dir.exists():
    model_files = list(models_dir.glob("*.pt"))
    if model_files:
        print("✅ Models found:")
        for model_file in model_files:
            size = model_file.stat().st_size / 1024  # KB
            mtime = time.ctime(model_file.stat().st_mtime)
            print(f"   - {model_file.name} ({size:.1f} KB) - Modified: {mtime}")
    else:
        print("⏳ No models saved yet (training in progress...)")
else:
    print("⏳ Models directory not created yet (training starting...)")

print()

if checkpoints_dir.exists():
    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_round_*.pt"))
    if checkpoint_files:
        print(f"✅ Training checkpoints: {len(checkpoint_files)} rounds completed")
        latest = checkpoint_files[-1]
        print(f"   Latest: {latest.name}")
    else:
        print("⏳ No checkpoints yet (training in progress...)")
else:
    print("⏳ Checkpoints directory not created yet")

print()

# Check data partitions
partitions_dir = Path("data/partitions")
if partitions_dir.exists():
    partition_files = list(partitions_dir.glob("client_*_train.pt"))
    print(f"✅ Data partitions ready: {len(partition_files) // 2} clients")
else:
    print("❌ Data partitions not found!")

print()
print("="*60)
print()

# Instructions
print("📋 What's happening:")
print()
print("1. Server is running on localhost:8080")
print("2. Three clients are connecting and training")
print("3. Training will run for 10 rounds (~5-10 minutes)")
print("4. Model will be saved to: models/best_model.pt")
print()
print("💡 To monitor progress:")
print("   - Check the terminal windows that opened")
print("   - Look for 'Round X - Training Results' messages")
print("   - Wait for 'Best model saved' message")
print()
print("⏱️  Expected completion time: 5-10 minutes")
print()
print("📊 After training completes:")
print("   1. Refresh dashboard (F5 in browser)")
print("   2. View model performance and explanations")
print("   3. Make predictions on new patients")
print()
print("="*60)
