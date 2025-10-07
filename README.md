# Privacy-Preserving Federated Learning with Explainable AI for Disease Prediction

A production-ready federated learning system for early disease prediction that combines privacy-preserving techniques with explainable AI for clinical interpretability.

## 🎯 Project Overview

This project implements a federated learning pipeline that:
- **Preserves Privacy**: Uses differential privacy and secure aggregation
- **Explains Predictions**: Generates patient-level and global explanations via SHAP and LIME
- **Simulates Real Hospitals**: Partitions data across multiple heterogeneous clients
- **Predicts Disease Risk**: Focuses on cardiovascular and diabetes risk prediction

## 🏗️ Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Client 1   │  │  Client 2   │  │  Client 3   │
│  (Hospital) │  │  (Hospital) │  │  (Hospital) │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       │    Model Updates (Encrypted)    │
       └────────────┬───────────────────┘
                    │
            ┌───────▼────────┐
            │  FL Server     │
            │  (Aggregator)  │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │  Global Model  │
            │  + XAI Layer   │
            └────────────────┘
```

## 📁 Project Structure

```
/ai_disease_prediction/
│
├── data/                       # Dataset storage
│   ├── raw/                   # Original datasets
│   ├── processed/             # Preprocessed data
│   └── partitions/            # Client-specific splits
│
├── federated/                 # Federated learning core
│   ├── __init__.py
│   ├── client.py             # Client-side training logic
│   ├── server.py             # Server aggregation logic
│   ├── models.py             # Neural network architectures
│   └── config.py             # FL configuration
│
├── privacy/                   # Privacy mechanisms
│   ├── __init__.py
│   ├── differential_privacy.py  # DP implementation
│   └── secure_aggregation.py   # Encrypted aggregation
│
├── explainability/            # XAI modules
│   ├── __init__.py
│   ├── shap_explainer.py     # SHAP-based explanations
│   ├── lime_explainer.py     # LIME-based explanations
│   └── visualizations.py     # Explanation plots
│
├── preprocessing/             # Data preparation
│   ├── __init__.py
│   ├── data_loader.py        # Load datasets
│   ├── partitioner.py        # Split data across clients
│   └── preprocessor.py       # Normalization, encoding
│
├── evaluation/                # Model evaluation
│   ├── __init__.py
│   ├── metrics.py            # Accuracy, F1, AUROC
│   └── stability.py          # Explanation stability
│
├── dashboard/                 # Interactive UI
│   ├── app.py                # Streamlit dashboard
│   └── components/           # UI components
│
├── tests/                     # Unit and integration tests
│   ├── test_model.py
│   ├── test_privacy.py
│   └── test_explanations.py
│
├── configs/                   # Configuration files
│   ├── fl_config.yaml        # FL hyperparameters
│   └── privacy_config.yaml   # Privacy budgets
│
├── notebooks/                 # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai_disease_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download and partition datasets
python preprocessing/data_loader.py --download
python preprocessing/partitioner.py --num-clients 3 --heterogeneous
```

### 3. Run Federated Training

```bash
# Start FL server
python federated/server.py --rounds 50 --min-clients 3

# In separate terminals, start clients
python federated/client.py --client-id 1
python federated/client.py --client-id 2
python federated/client.py --client-id 3
```

### 4. Generate Explanations

```bash
# Generate SHAP and LIME explanations
python explainability/shap_explainer.py --model-path models/global_model.pth
python explainability/lime_explainer.py --model-path models/global_model.pth
```

### 5. Launch Dashboard

```bash
# Start interactive dashboard
streamlit run dashboard/app.py
```

## 🔬 Key Features

### Federated Learning
- **Framework**: Flower (flexible and scalable)
- **Aggregation**: FedAvg with weighted averaging
- **Client Simulation**: Heterogeneous data distribution across 3+ clients
- **Communication**: Secure model weight exchange

### Privacy Preservation
- **Differential Privacy**: Opacus integration with configurable ε and δ
- **Secure Aggregation**: Encrypted gradient aggregation
- **Privacy Budget Tracking**: Monitor cumulative privacy loss
- **Noise Injection**: Gaussian mechanism for gradient perturbation

### Explainable AI
- **SHAP**: Global feature importance and patient-level explanations
- **LIME**: Local interpretable model-agnostic explanations
- **Stability Analysis**: Track explanation consistency across FL rounds
- **Clinical Interpretability**: Feature attribution with medical context

### Model Architecture
- **Base Model**: Multi-layer perceptron (MLP) classifier
- **Input Features**: Clinical variables (age, BP, glucose, etc.)
- **Output**: Disease risk probability (0-1)
- **Optimization**: Adam optimizer with learning rate scheduling

## 📊 Datasets

### UCI Heart Disease Dataset
- **Samples**: 303 patients
- **Features**: 13 clinical attributes
- **Target**: Heart disease presence (binary)
- **Source**: UCI ML Repository

### UCI Diabetes Dataset
- **Samples**: 768 patients
- **Features**: 8 clinical measurements
- **Target**: Diabetes diagnosis (binary)
- **Source**: UCI ML Repository

## 🔒 Privacy Guarantees

- **Differential Privacy**: (ε=1.0, δ=1e-5) by default
- **Gradient Clipping**: L2 norm clipping at threshold 1.0
- **Noise Mechanism**: Gaussian noise calibrated to privacy budget
- **Secure Aggregation**: Homomorphic encryption for weight aggregation

## 📈 Evaluation Metrics

### Model Performance
- Accuracy
- Precision, Recall, F1-Score
- AUROC (Area Under ROC Curve)
- Confusion Matrix

### Explanation Quality
- Feature Importance Consistency
- Explanation Stability Across Rounds
- Clinical Relevance Score

### Privacy Metrics
- Privacy Budget Consumption
- Utility-Privacy Trade-off Analysis

## 🛠️ Configuration

Edit `configs/fl_config.yaml` to customize:

```yaml
federated:
  num_rounds: 50
  min_clients: 3
  fraction_fit: 1.0
  fraction_evaluate: 1.0

model:
  hidden_layers: [64, 32, 16]
  activation: relu
  dropout: 0.3

training:
  batch_size: 32
  local_epochs: 5
  learning_rate: 0.001

privacy:
  enable_dp: true
  epsilon: 1.0
  delta: 1e-5
  max_grad_norm: 1.0
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_model.py -v
pytest tests/test_privacy.py -v
pytest tests/test_explanations.py -v

# Check coverage
pytest --cov=federated --cov=privacy --cov=explainability
```

## 📚 Documentation

- **API Documentation**: See `docs/api.md`
- **Tutorial Notebooks**: Check `notebooks/`
- **Architecture Guide**: See `docs/architecture.md`
- **Privacy Analysis**: See `docs/privacy_analysis.md`

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Flower**: Federated learning framework
- **Opacus**: Differential privacy library
- **SHAP**: Explainability framework
- **UCI ML Repository**: Dataset source

## 📧 Contact

For questions or collaboration: [your-email@example.com]

## 🗺️ Roadmap

- [x] Phase 1: Centralized baseline
- [x] Phase 2: Federated simulation
- [x] Phase 3: Privacy integration
- [x] Phase 4: Explainability layer
- [ ] Phase 5: Real hospital deployment
- [ ] Phase 6: Regulatory compliance (HIPAA, GDPR)
- [ ] Phase 7: Real-time inference API
- [ ] Phase 8: Clinician feedback loop

---

**Built with ❤️ for privacy-preserving healthcare AI**
