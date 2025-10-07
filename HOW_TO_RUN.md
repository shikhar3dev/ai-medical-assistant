# 🚀 How to Run the Federated Learning System

## ✅ Setup Complete!

Your data has been downloaded and partitioned. You're ready to train!

## 🎯 Quick Start (3 Easy Steps)

### Step 1: Start the FL Server

**Double-click**: `start_server.bat`

Or in terminal:
```cmd
cd c:\ai_disease_prediction
start_server.bat
```

**What you'll see**:
```
Starting Flower server on localhost:8080
Waiting for 3 clients...
```

### Step 2: Start 3 Clients (in separate windows)

**Double-click each**:
1. `start_client0.bat`
2. `start_client1.bat`  
3. `start_client2.bat`

Or in 3 separate terminals:
```cmd
# Terminal 2
cd c:\ai_disease_prediction
start_client0.bat

# Terminal 3
cd c:\ai_disease_prediction
start_client1.bat

# Terminal 4
cd c:\ai_disease_prediction
start_client2.bat
```

**What you'll see**:
```
[Client 0] Initialized with 140 training samples
[Client 0] Connecting to server at localhost:8080
[Client 0] Train Loss: 0.6234, Accuracy: 0.6500
```

### Step 3: Watch Training Progress

The server will show:
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
Global Metrics: Loss=0.5985, Acc=0.6900
============================================================
```

Training will run for **10 rounds** (about 5-10 minutes).

## 📊 After Training

### Generate Explanations

```cmd
cd c:\ai_disease_prediction
venv\Scripts\activate
set PYTHONPATH=%CD%;%PYTHONPATH%

python explainability\shap_explainer.py --num-samples 100
python explainability\lime_explainer.py --num-instances 10
```

### Launch Dashboard

**Double-click**: `start_dashboard.bat`

Or:
```cmd
cd c:\ai_disease_prediction
start_dashboard.bat
```

The dashboard will open at: **http://localhost:8501**

## 📁 Files Created

After training, you'll have:

```
data/
├── raw/
│   ├── heart_disease.csv       # Downloaded dataset
│   └── diabetes.csv
├── processed/
│   ├── preprocessor.pkl        # Data preprocessor
│   └── test_data.pt           # Test set
└── partitions/
    ├── client_0_train.pt       # Client 0 data
    ├── client_0_val.pt
    ├── client_1_train.pt       # Client 1 data
    ├── client_1_val.pt
    ├── client_2_train.pt       # Client 2 data
    └── client_2_val.pt

models/
└── best_model.pt              # Trained model

checkpoints/
└── checkpoint_round_*.pt      # Saved checkpoints

results/
└── explanations/
    ├── feature_importance.csv
    ├── shap_explanations.pkl
    └── lime_*.html
```

## 🎮 Batch Files Reference

| File | Purpose |
|------|---------|
| `setup_and_run.bat` | Initial setup (run once) |
| `start_server.bat` | Start FL server |
| `start_client0.bat` | Start client 0 |
| `start_client1.bat` | Start client 1 |
| `start_client2.bat` | Start client 2 |
| `start_dashboard.bat` | Launch dashboard |

## 📈 Expected Results

After 10 rounds:
- **Accuracy**: ~0.75-0.80
- **AUROC**: ~0.80-0.85
- **Training time**: 5-10 minutes

After 50 rounds (for better results):
- **Accuracy**: ~0.82-0.85
- **AUROC**: ~0.87-0.90
- **Training time**: 20-30 minutes

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'federated'"

**Solution**: The batch files now automatically set PYTHONPATH. If you're running manually:
```cmd
set PYTHONPATH=%CD%;%PYTHONPATH%
```

### Issue: "Port 8080 already in use"

**Solution**: Edit `configs\fl_config.yaml`:
```yaml
federated:
  server_address: "localhost:8081"
```

### Issue: Clients can't connect to server

**Solution**: 
1. Make sure server is running first
2. Check that all terminals are in `c:\ai_disease_prediction`
3. Verify firewall isn't blocking port 8080

### Issue: Training is slow

**Solution**: 
- Reduce number of rounds: Edit `start_server.bat` and change `--rounds 10` to `--rounds 5`
- Reduce batch size: Edit `configs\fl_config.yaml` and set `batch_size: 16`

## 🎯 Training Options

### Quick Test (5 rounds)
Edit `start_server.bat`:
```batch
python federated\server.py --rounds 5 --min-clients 3
```

### Full Training (50 rounds)
Edit `start_server.bat`:
```batch
python federated\server.py --rounds 50 --min-clients 3
```

### Different Dataset
Run setup again with diabetes:
```cmd
set PYTHONPATH=%CD%;%PYTHONPATH%
venv\Scripts\activate
python preprocessing\partitioner.py --num-clients 3 --dataset diabetes
```

## 📊 Dashboard Features

Once training is complete, the dashboard shows:

1. **Overview Page**
   - System configuration
   - Quick performance metrics
   - Model architecture

2. **Model Performance**
   - Accuracy, AUROC, F1-Score
   - Confusion matrix
   - ROC curves
   - Prediction distribution

3. **Explainability**
   - Feature importance (SHAP)
   - Individual predictions
   - Top contributing features

4. **Privacy Analysis**
   - Privacy budget tracking
   - ε (epsilon) consumption
   - Privacy parameters

5. **Patient Prediction**
   - Enter patient data
   - Get risk prediction
   - See risk gauge

## 🚀 Next Steps

1. ✅ **Training Complete** - Model saved to `models/best_model.pt`
2. ✅ **Generate Explanations** - Run SHAP and LIME
3. ✅ **Launch Dashboard** - Visualize results
4. 📊 **Analyze Results** - Check feature importance
5. 🔬 **Experiment** - Try different configurations

## 💡 Tips

- **Keep terminals open** during training
- **Server must start first** before clients
- **Wait for "Waiting for clients"** message before starting clients
- **All 3 clients must connect** before training begins
- **Don't close terminals** until training completes

## 📚 More Information

- **Full Documentation**: [README.md](README.md)
- **Configuration Guide**: [configs/fl_config.yaml](configs/fl_config.yaml)
- **Project Overview**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Quick Reference**: [QUICKSTART.md](QUICKSTART.md)

---

**🎉 Happy Training!** Your federated learning system is ready to go!
