# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-06

### Added

#### Core Features
- **Federated Learning Framework**: Implemented FL using Flower
  - Server-side aggregation with FedAvg strategy
  - Client-side training with local model updates
  - Support for heterogeneous data distribution
  - Configurable number of clients and rounds

#### Privacy Mechanisms
- **Differential Privacy**: Integrated Opacus for DP
  - Gradient clipping with configurable threshold
  - Gaussian noise injection
  - Privacy budget tracking (epsilon, delta)
  - Per-round privacy accounting
- **Secure Aggregation**: Basic encrypted aggregation support
  - Simple masking-based aggregation
  - Placeholder for homomorphic encryption

#### Explainability
- **SHAP Integration**: Global and local explanations
  - DeepExplainer, GradientExplainer, KernelExplainer
  - Feature importance ranking
  - Instance-level explanations
  - Waterfall plots and summary plots
- **LIME Integration**: Local interpretable explanations
  - Tabular explainer for medical data
  - Feature contribution analysis
  - HTML export for explanations
  - Comparison with SHAP

#### Data Processing
- **Dataset Support**: UCI Heart Disease and Diabetes datasets
  - Automatic download from UCI repository
  - Synthetic data generation fallback
  - Data preprocessing and normalization
  - Class imbalance handling (SMOTE, oversampling)
- **Data Partitioning**: Multiple strategies
  - IID (Independent and Identically Distributed)
  - Heterogeneous (Dirichlet distribution)
  - Pathological (extreme non-IID)

#### Model Architecture
- **Neural Network**: Multi-layer perceptron
  - Configurable hidden layers
  - Batch normalization support
  - Dropout regularization
  - Multiple activation functions (ReLU, Tanh, Sigmoid)
- **Loss Functions**: Multiple options
  - Binary Cross-Entropy
  - Focal Loss for imbalance
  - Weighted BCE

#### Evaluation
- **Metrics**: Comprehensive evaluation
  - Accuracy, Precision, Recall, F1-Score
  - AUROC, AUPRC
  - Sensitivity, Specificity
  - Confusion Matrix
- **Visualization**: Multiple plots
  - ROC curves
  - Precision-Recall curves
  - Calibration curves
  - Training history plots
- **Stability Tracking**: Explanation consistency
  - Feature importance stability across rounds
  - Rank correlation analysis
  - Jaccard similarity metrics

#### Dashboard
- **Streamlit Dashboard**: Interactive web interface
  - Model performance visualization
  - Feature importance plots
  - Individual prediction explanations
  - Privacy budget monitoring
  - Patient risk prediction interface

#### Configuration
- **YAML Configuration**: Flexible config system
  - FL parameters (rounds, clients, strategy)
  - Model hyperparameters
  - Privacy settings
  - Data processing options
  - Explainability settings

#### Testing
- **Unit Tests**: Core functionality tests
  - Model architecture tests
  - Privacy mechanism tests
  - Explainability tests
  - Data processing tests

#### Documentation
- **Comprehensive Documentation**:
  - README with full project overview
  - Quick Start guide
  - Contributing guidelines
  - API documentation
  - Configuration examples

### Project Structure
```
ai_disease_prediction/
├── configs/              # Configuration files
├── data/                 # Dataset storage
├── dashboard/            # Streamlit dashboard
├── evaluation/           # Evaluation metrics and tools
├── explainability/       # SHAP and LIME explainers
├── federated/            # FL client and server
├── preprocessing/        # Data loading and partitioning
├── privacy/              # DP and secure aggregation
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
└── README.md            # Documentation
```

### Dependencies
- PyTorch >= 2.0.0
- Flower >= 1.5.0
- Opacus >= 1.4.0
- SHAP >= 0.42.0
- LIME >= 0.2.0
- Streamlit >= 1.25.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0

### Known Limitations
- Homomorphic encryption is placeholder only
- No real-time inference API yet
- Limited to binary classification
- No multi-modal data support
- Dashboard requires model to be pre-trained

### Future Roadmap
- [ ] Real hospital data integration
- [ ] Multi-class classification support
- [ ] Advanced privacy mechanisms (HE, MPC)
- [ ] Real-time inference API
- [ ] Regulatory compliance tools (HIPAA, GDPR)
- [ ] Clinician feedback integration
- [ ] Mobile app for predictions
- [ ] Cloud deployment support

---

## [Unreleased]

### Planned Features
- Docker containerization
- Kubernetes deployment configs
- Advanced model architectures (Transformers)
- Federated transfer learning
- Cross-silo and cross-device FL
- Byzantine-robust aggregation
- Personalized federated learning
- Federated hyperparameter tuning

---

**Note**: This is the initial release. Future versions will include bug fixes, performance improvements, and new features based on community feedback.
