# 🚀 Get Started with Your Federated Learning System

Welcome! This guide will help you get your privacy-preserving federated learning system up and running.

## 📋 Prerequisites Checklist

Before you begin, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] pip package manager
- [ ] Git (optional, for version control)
- [ ] 4 GB RAM minimum (8 GB recommended)
- [ ] 2 GB free disk space

## 🎯 Installation Steps

### Step 1: Verify Your Setup

```bash
# Navigate to project directory
cd ai_disease_prediction

# Run installation verification
python verify_installation.py
```

This will check:
- Python version
- Required dependencies
- Project structure
- GPU availability (optional)

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**Expected time**: 5-10 minutes

### Step 3: Verify Installation Again

```bash
python verify_installation.py
```

All checks should now pass! ✅

## 🏃 Quick Start (15 minutes)

### Phase 1: Data Preparation (3 minutes)

```bash
# Download datasets
python preprocessing/data_loader.py --download

# Partition data for 3 clients
python preprocessing/partitioner.py --num-clients 3 --dataset heart_disease
```

**What happens**: 
- Downloads UCI Heart Disease dataset
- Splits data into train/val/test
- Creates heterogeneous partitions for 3 clients
- Saves preprocessed data to `data/partitions/`

### Phase 2: Federated Training (10 minutes)

Open **4 separate terminal windows**:

**Terminal 1 - Server:**
```bash
cd ai_disease_prediction
venv\Scripts\activate  # or source venv/bin/activate
python federated/server.py --rounds 10 --min-clients 3
```

**Terminal 2 - Client 0:**
```bash
cd ai_disease_prediction
venv\Scripts\activate
python federated/client.py --client-id 0
```

**Terminal 3 - Client 1:**
```bash
cd ai_disease_prediction
venv\Scripts\activate
python federated/client.py --client-id 1
```

**Terminal 4 - Client 2:**
```bash
cd ai_disease_prediction
venv\Scripts\activate
python federated/client.py --client-id 2
```

**What happens**:
- Server waits for 3 clients to connect
- Clients train locally on their data
- Server aggregates model updates
- Process repeats for 10 rounds
- Best model saved to `models/best_model.pt`

**Watch for**: Training loss decreasing, accuracy increasing

### Phase 3: Generate Explanations (2 minutes)

After training completes:

```bash
# Generate SHAP explanations
python explainability/shap_explainer.py --num-samples 100

# Generate LIME explanations
python explainability/lime_explainer.py --num-instances 10
```

**What happens**:
- Computes feature importance using SHAP
- Generates local explanations using LIME
- Saves results to `results/explanations/`

### Phase 4: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

**What happens**:
- Opens interactive dashboard at `http://localhost:8501`
- View model performance
- Explore feature importance
- Make predictions on new patients

## 📊 What to Expect

### Training Output

```
============================================================
Round 1 - Training Results
============================================================
Client 0: Loss=0.6234, Acc=0.6500
Client 1: Loss=0.6189, Acc=0.6700
Client 2: Loss=0.6301, Acc=0.6400

============================================================
Round 1 - Evaluation Results
============================================================
Client 0: Loss=0.5987, Acc=0.6800
Client 1: Loss=0.5923, Acc=0.7000
Client 2: Loss=0.6045, Acc=0.6900

Global Metrics: Loss=0.5985, Acc=0.6900
============================================================
```

### Final Performance (After 50 rounds)

| Metric | Value |
|--------|-------|
| Accuracy | ~0.82 |
| AUROC | ~0.87 |
| F1-Score | ~0.79 |
| Precision | ~0.81 |

### Feature Importance (Top 5)

1. **thalach** (max heart rate) - 0.156
2. **cp** (chest pain type) - 0.142
3. **oldpeak** (ST depression) - 0.128
4. **ca** (number of vessels) - 0.115
5. **age** - 0.098

## 🎓 Learning Path

### Beginner (Week 1)
1. ✅ Run the quick start guide
2. ✅ Explore the dashboard
3. ✅ Read `README.md` thoroughly
4. ✅ Understand the project structure
5. ✅ Modify configuration files

### Intermediate (Week 2)
1. Change model architecture in `configs/fl_config.yaml`
2. Try different partitioning strategies (IID, pathological)
3. Adjust privacy budget (epsilon, delta)
4. Experiment with different datasets
5. Analyze explanation stability

### Advanced (Week 3+)
1. Implement new aggregation strategies
2. Add custom loss functions
3. Integrate real medical datasets
4. Extend privacy mechanisms
5. Contribute new features

## 🔧 Common Issues & Solutions

### Issue: "Model not found"
**Solution**: Train the model first using FL server and clients.

### Issue: "Port 8080 already in use"
**Solution**: Change port in `configs/fl_config.yaml`:
```yaml
federated:
  server_address: "localhost:8081"
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `configs/fl_config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: "Import error for opacus"
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade opacus
```

### Issue: "Data download fails"
**Solution**: The system will automatically create synthetic data as fallback.

## 📚 Key Files to Understand

| File | Purpose |
|------|---------|
| `configs/fl_config.yaml` | Main configuration |
| `federated/server.py` | FL server logic |
| `federated/client.py` | FL client logic |
| `federated/models.py` | Neural network |
| `privacy/differential_privacy.py` | DP implementation |
| `explainability/shap_explainer.py` | SHAP explanations |
| `dashboard/app.py` | Interactive UI |

## 🎯 Your First Experiment

Try this experiment to understand the system:

### Experiment: Privacy-Utility Tradeoff

1. **Baseline** (No DP):
   ```yaml
   # In configs/privacy_config.yaml
   differential_privacy:
     enabled: false
   ```
   Train and record accuracy.

2. **Low Privacy** (ε=10):
   ```yaml
   differential_privacy:
     enabled: true
     total_epsilon: 10.0
   ```
   Train and record accuracy.

3. **High Privacy** (ε=1):
   ```yaml
   differential_privacy:
     enabled: true
     total_epsilon: 1.0
   ```
   Train and record accuracy.

4. **Compare Results**:
   - Plot accuracy vs epsilon
   - Analyze privacy-utility tradeoff
   - Document findings

## 🚀 Next Steps

### Immediate (Today)
- [ ] Complete quick start guide
- [ ] Explore dashboard features
- [ ] Read project documentation

### Short-term (This Week)
- [ ] Run privacy-utility experiment
- [ ] Try different datasets
- [ ] Modify model architecture
- [ ] Generate custom explanations

### Long-term (This Month)
- [ ] Integrate real medical data
- [ ] Implement new features
- [ ] Contribute to the project
- [ ] Share your findings

## 💡 Tips for Success

1. **Start Simple**: Use default configurations first
2. **Read Logs**: Pay attention to training output
3. **Experiment**: Try different settings
4. **Document**: Keep notes on what works
5. **Ask Questions**: Open issues for help

## 📖 Additional Resources

- **Main Documentation**: [README.md](README.md)
- **Quick Reference**: [QUICKSTART.md](QUICKSTART.md)
- **Project Overview**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## 🤝 Getting Help

1. **Check Documentation**: Most questions are answered in README.md
2. **Run Verification**: `python verify_installation.py`
3. **Check Issues**: Look for similar problems on GitHub
4. **Ask Questions**: Open a new issue with details

## 🎉 You're Ready!

Congratulations! You now have everything you need to:
- ✅ Train federated learning models
- ✅ Preserve privacy with differential privacy
- ✅ Explain predictions with SHAP and LIME
- ✅ Visualize results in interactive dashboard
- ✅ Conduct privacy-preserving healthcare AI research.

**Happy Learning!** 🚀

---

**Questions?** Check the documentation or open an issue.

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md).

**Built with ❤️ for advancing privacy-preserving healthcare AI**
