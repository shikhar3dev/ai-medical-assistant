"""Verify Installation and Setup."""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'flwr': 'Flower',
        'opacus': 'Opacus',
        'shap': 'SHAP',
        'lime': 'LIME',
        'streamlit': 'Streamlit',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yaml': 'PyYAML',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            all_installed = False
    
    return all_installed


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'configs',
        'federated',
        'privacy',
        'preprocessing',
        'explainability',
        'evaluation',
        'dashboard',
        'tests'
    ]
    
    required_files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'configs/fl_config.yaml',
        'configs/privacy_config.yaml'
    ]
    
    all_present = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - Missing")
            all_present = False
    
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - Missing")
            all_present = False
    
    return all_present


def check_data_setup():
    """Check if data directories exist."""
    print("\nChecking data setup...")
    
    data_dirs = ['data', 'data/raw', 'data/processed', 'data/partitions']
    
    for dir_name in data_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ - Not created yet (will be created automatically)")
    
    return True


def check_gpu_availability():
    """Check if GPU is available."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {device_name}")
            return True
        else:
            print("⚠️  No GPU available (CPU will be used)")
            return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def run_quick_test():
    """Run a quick functionality test."""
    print("\nRunning quick functionality test...")
    
    try:
        import torch
        from federated.models import DiseasePredictor
        
        # Create a simple model
        model = DiseasePredictor(input_size=10, hidden_layers=[32, 16])
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        
        if output.shape == (5, 1):
            print("✅ Model creation and forward pass")
            return True
        else:
            print("❌ Model output shape incorrect")
            return False
    
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "="*60)
    print("INSTALLATION VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Read QUICKSTART.md for usage instructions")
        print("2. Download data: python preprocessing/data_loader.py --download")
        print("3. Partition data: python preprocessing/partitioner.py --num-clients 3")
        print("4. Start training with FL server and clients")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print()


def main():
    """Main verification function."""
    print("="*60)
    print("FEDERATED LEARNING INSTALLATION VERIFICATION")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Structure': check_project_structure(),
        'Data Setup': check_data_setup(),
        'GPU Availability': check_gpu_availability(),
        'Functionality Test': run_quick_test()
    }
    
    print_summary(results)


if __name__ == "__main__":
    main()
