# 📚 Project Documentation Index

Quick navigation to all project documentation and resources.

## 🎯 Getting Started

| Document | Description | Time Required |
|----------|-------------|---------------|
| **[GET_STARTED.md](GET_STARTED.md)** | Complete beginner's guide | 15 min |
| **[QUICKSTART.md](QUICKSTART.md)** | Quick reference guide | 5 min |
| **[README.md](README.md)** | Full project documentation | 20 min |

**Start here**: [GET_STARTED.md](GET_STARTED.md) → [QUICKSTART.md](QUICKSTART.md) → [README.md](README.md)

## 📖 Core Documentation

### Overview & Architecture
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete project overview, architecture, and features
- **[README.md](README.md)** - Main documentation with detailed explanations
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes

### Setup & Installation
- **[GET_STARTED.md](GET_STARTED.md)** - Step-by-step installation and first run
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference for common tasks
- **[verify_installation.py](verify_installation.py)** - Installation verification script

### Development
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines and workflow
- **[LICENSE](LICENSE)** - MIT License terms
- **[.env.example](.env.example)** - Environment variables template

## 🗂️ Code Documentation

### Configuration Files
```
configs/
├── fl_config.yaml          # Federated learning settings
└── privacy_config.yaml     # Privacy parameters
```

**What to configure**:
- Number of FL rounds
- Client settings
- Model architecture
- Privacy budget (ε, δ)
- Learning rate and optimization

### Core Modules

#### 1. Federated Learning (`federated/`)
- **`models.py`** - Neural network architectures (DiseasePredictor, loss functions)
- **`client.py`** - FL client implementation (local training, DP integration)
- **`server.py`** - FL server implementation (aggregation, model saving)

#### 2. Privacy (`privacy/`)
- **`differential_privacy.py`** - DP mechanisms (gradient clipping, noise injection)
- **`secure_aggregation.py`** - Encrypted aggregation (masking, HE placeholders)

#### 3. Data Processing (`preprocessing/`)
- **`data_loader.py`** - Dataset loading (UCI Heart Disease, Diabetes)
- **`preprocessor.py`** - Data preprocessing (scaling, SMOTE, normalization)
- **`partitioner.py`** - Client data partitioning (IID, heterogeneous, pathological)

#### 4. Explainability (`explainability/`)
- **`shap_explainer.py`** - SHAP explanations (global, local, feature importance)
- **`lime_explainer.py`** - LIME explanations (instance-level, HTML export)
- **`visualizations.py`** - Explanation plots (importance, stability, comparison)

#### 5. Evaluation (`evaluation/`)
- **`metrics.py`** - Performance metrics (accuracy, AUROC, confusion matrix)
- **`stability.py`** - Explanation stability tracking across FL rounds

#### 6. Dashboard (`dashboard/`)
- **`app.py`** - Streamlit interactive dashboard (5 pages: overview, performance, explainability, privacy, prediction)

### Testing (`tests/`)
- **`test_model.py`** - Model architecture tests
- **`test_privacy.py`** - Privacy mechanism tests
- **`test_explanations.py`** - Explainability tests

## 🚀 Quick Command Reference

### Installation
```bash
pip install -r requirements.txt
python verify_installation.py
```

### Data Preparation
```bash
python preprocessing/data_loader.py --download
python preprocessing/partitioner.py --num-clients 3
```

### Training
```bash
# Terminal 1 - Server
python federated/server.py --rounds 50

# Terminals 2-4 - Clients
python federated/client.py --client-id 0
python federated/client.py --client-id 1
python federated/client.py --client-id 2
```

### Explanations
```bash
python explainability/shap_explainer.py --num-samples 100
python explainability/lime_explainer.py --num-instances 10
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

### Testing
```bash
pytest tests/ -v
pytest tests/test_model.py -v
```

### Experiment Runner
```bash
python run_experiment.py --num-clients 3 --rounds 10
```

## 📊 Project Structure

```
ai_disease_prediction/
│
├── 📄 Documentation
│   ├── README.md              # Main documentation
│   ├── GET_STARTED.md         # Beginner's guide
│   ├── QUICKSTART.md          # Quick reference
│   ├── PROJECT_SUMMARY.md     # Project overview
│   ├── CONTRIBUTING.md        # Contribution guide
│   ├── CHANGELOG.md           # Version history
│   ├── LICENSE                # MIT License
│   └── INDEX.md               # This file
│
├── ⚙️ Configuration
│   └── configs/
│       ├── fl_config.yaml     # FL settings
│       └── privacy_config.yaml # Privacy settings
│
├── 🧠 Core Modules
│   ├── federated/             # FL implementation
│   ├── privacy/               # Privacy mechanisms
│   ├── preprocessing/         # Data processing
│   ├── explainability/        # XAI modules
│   ├── evaluation/            # Metrics & evaluation
│   └── dashboard/             # Interactive UI
│
├── 🧪 Testing
│   └── tests/                 # Unit tests
│
├── 🔧 Utilities
│   ├── run_experiment.py      # Experiment runner
│   ├── verify_installation.py # Setup verification
│   ├── requirements.txt       # Dependencies
│   └── setup.py              # Package setup
│
└── 📁 Data & Results
    ├── data/                  # Datasets
    ├── models/                # Trained models
    ├── results/               # Outputs
    └── logs/                  # Training logs
```

## 🎓 Learning Path

### Level 1: Beginner (Week 1)
**Goal**: Understand the system and run basic experiments

1. Read [GET_STARTED.md](GET_STARTED.md)
2. Run installation verification
3. Complete quick start guide
4. Explore dashboard
5. Read [README.md](README.md) sections 1-4

**Time**: 2-3 hours

### Level 2: Intermediate (Week 2)
**Goal**: Customize and experiment

1. Modify `configs/fl_config.yaml`
2. Try different datasets
3. Adjust privacy settings
4. Analyze explanation stability
5. Read code in `federated/` and `privacy/`

**Time**: 5-7 hours

### Level 3: Advanced (Week 3+)
**Goal**: Extend and contribute

1. Implement new features
2. Add custom aggregation strategies
3. Integrate real datasets
4. Write tests
5. Contribute to the project

**Time**: 10+ hours

## 🔍 Find What You Need

### I want to...

**...get started quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...understand the full system**
→ [README.md](README.md) + [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**...configure the system**
→ `configs/fl_config.yaml` + `configs/privacy_config.yaml`

**...modify the model**
→ `federated/models.py` + `configs/fl_config.yaml`

**...adjust privacy settings**
→ `configs/privacy_config.yaml` + `privacy/differential_privacy.py`

**...add new features**
→ [CONTRIBUTING.md](CONTRIBUTING.md)

**...understand explanations**
→ `explainability/shap_explainer.py` + `explainability/lime_explainer.py`

**...run experiments**
→ `run_experiment.py` + [QUICKSTART.md](QUICKSTART.md)

**...troubleshoot issues**
→ [GET_STARTED.md](GET_STARTED.md) section "Common Issues"

**...contribute code**
→ [CONTRIBUTING.md](CONTRIBUTING.md)

## 📚 External Resources

### Federated Learning
- [Flower Documentation](https://flower.dev/docs/)
- [Federated Learning Book](https://www.federated-learning.org/)

### Differential Privacy
- [Opacus Tutorials](https://opacus.ai/tutorials/)
- [DP Book](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

### Explainable AI
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME GitHub](https://github.com/marcotcr/lime)

### Medical AI
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Healthcare AI Papers](https://arxiv.org/list/cs.LG/recent)

## 🆘 Getting Help

1. **Check documentation** - Most answers are here
2. **Run verification** - `python verify_installation.py`
3. **Search issues** - Look for similar problems
4. **Ask questions** - Open a new issue with details

## 📝 Quick Tips

- **Always activate virtual environment** before running commands
- **Read error messages carefully** - they usually tell you what's wrong
- **Start with default settings** - customize after understanding basics
- **Keep notes** - document what works and what doesn't
- **Experiment** - try different configurations

## 🎯 Common Workflows

### Workflow 1: First Time Setup
```
1. Install dependencies → verify_installation.py
2. Download data → data_loader.py
3. Partition data → partitioner.py
4. Train model → server.py + client.py
5. Generate explanations → shap_explainer.py
6. View results → dashboard/app.py
```

### Workflow 2: New Experiment
```
1. Modify configs/fl_config.yaml
2. Run training
3. Generate explanations
4. Compare results in dashboard
```

### Workflow 3: Development
```
1. Create feature branch
2. Write code + tests
3. Run pytest
4. Format with black
5. Submit pull request
```

## ✅ Checklist for Success

- [ ] Read GET_STARTED.md
- [ ] Install all dependencies
- [ ] Verify installation
- [ ] Download and partition data
- [ ] Train model successfully
- [ ] Generate explanations
- [ ] Explore dashboard
- [ ] Read README.md thoroughly
- [ ] Try modifying configurations
- [ ] Run tests

## 🎉 You're All Set!

You now have a complete reference to navigate the entire project. Happy learning and experimenting!

---

**Last Updated**: 2025-10-06

**Questions?** Check the documentation or open an issue.

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md).
