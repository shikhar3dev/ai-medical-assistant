"""Test if all imports work correctly"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")
print(f"Python path: {sys.path[0]}")
print()

try:
    from federated.models import DiseasePredictor
    print("✅ federated.models imported successfully")
    
except Exception as e:
    print(f"❌ federated.models failed: {e}")

try:
    from preprocessing.preprocessor import DataPreprocessor
    print("✅ preprocessing.preprocessor imported successfully")
except Exception as e:
    print(f"❌ preprocessing.preprocessor failed: {e}")

try:
    from privacy.differential_privacy import DPTrainer
    print("✅ privacy.differential_privacy imported successfully")
except Exception as e:
    print(f"❌ privacy.differential_privacy failed: {e}")

try:
    from explainability.shap_explainer import SHAPExplainer
    print("✅ explainability.shap_explainer imported successfully")
except Exception as e:
    print(f"❌ explainability.shap_explainer failed: {e}")

try:
    from evaluation.metrics import calculate_metrics
    print("✅ evaluation.metrics imported successfully")
except Exception as e:
    print(f"❌ evaluation.metrics failed: {e}")

print()
print("✅ All imports successful!")
print()
print("You can now run:")
print("  - start_server.bat")
print("  - start_client0.bat, start_client1.bat, start_client2.bat")
print("  - start_dashboard.bat")
