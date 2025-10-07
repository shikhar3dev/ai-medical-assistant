# Quick Start Guide

Get your federated learning system up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

### 1. Clone and Setup

```bash
cd ai_disease_prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Prepare Data

```bash
# Download datasets
python preprocessing/data_loader.py --download

# Partition data for 3 clients
python preprocessing/partitioner.py --num-clients 3 --dataset heart_disease
```

## Running Federated Learning

### Option 1: Quick Test (Automated)

```bash
# Run the experiment script
python run_experiment.py --num-clients 3 --rounds 10 --dataset heart_disease
```

### Option 2: Manual Training (Recommended)

Open **4 separate terminals** and run:

**Terminal 1 - FL Server:**
```bash
python federated/server.py --rounds 50 --min-clients 3
```

**Terminal 2 - Client 0:**
```bash
python federated/client.py --client-id 0
```

**Terminal 3 - Client 1:**
```bash
python federated/client.py --client-id 1
```

**Terminal 4 - Client 2:**
```bash
python federated/client.py --client-id 2
```

The server will wait for all 3 clients to connect, then start training for 50 rounds.

## Generate Explanations

After training completes:

```bash
# Generate SHAP explanations
python explainability/shap_explainer.py --num-samples 100

# Generate LIME explanations
python explainability/lime_explainer.py --num-instances 10
```

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Verify Installation

Run tests to ensure everything is working:

```bash
pytest tests/ -v
```

## Expected Results

After training for 50 rounds, you should see:
- **Accuracy**: ~0.80-0.85
- **AUROC**: ~0.85-0.90
- **F1-Score**: ~0.75-0.82

## Troubleshooting

### Issue: "Model not found"
**Solution**: Train the model first using the FL server and clients.

### Issue: "Data not found"
**Solution**: Run the data preparation steps:
```bash
python preprocessing/data_loader.py --download
python preprocessing/partitioner.py --num-clients 3
```

### Issue: "Port already in use"
**Solution**: Change the server address in `configs/fl_config.yaml`:
```yaml
federated:
  server_address: "localhost:8081"  # Change port
```

### Issue: Dependencies installation fails
**Solution**: Upgrade pip and try again:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Next Steps

1. **Customize Configuration**: Edit `configs/fl_config.yaml` to adjust:
   - Number of rounds
   - Privacy budget (epsilon, delta)
   - Model architecture
   - Learning rate

2. **Try Different Datasets**: Switch to diabetes dataset:
   ```bash
   python preprocessing/partitioner.py --num-clients 3 --dataset diabetes
   ```

3. **Adjust Privacy Settings**: Modify `configs/privacy_config.yaml` to:
   - Increase/decrease epsilon
   - Enable secure aggregation
   - Adjust gradient clipping

4. **Explore Dashboard**: Use the interactive dashboard to:
   - View model performance
   - Analyze feature importance
   - Explain individual predictions
   - Monitor privacy budget

## Common Commands

```bash
# Download data
python preprocessing/data_loader.py --download

# Partition data
python preprocessing/partitioner.py --num-clients 3

# Start server
python federated/server.py --rounds 50

# Start client
python federated/client.py --client-id 0

# Generate explanations
python explainability/shap_explainer.py

# Launch dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v
```

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review configuration files in `configs/`
- Examine example notebooks in `notebooks/`
- Open an issue on GitHub for bugs or questions

---

**Ready to go!** 🚀 Start with the manual training option for the best experience.
