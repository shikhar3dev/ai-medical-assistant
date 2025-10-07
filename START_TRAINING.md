# 🚀 START TRAINING NOW

## ✅ Dashboard is Working!

Your dashboard loaded successfully. Now you just need to train the model.

## 🎯 Quick Start Training (Choose One Method)

### Method 1: Double-Click Batch Files (EASIEST)

**Step 1**: Double-click **`start_server.bat`**
- Wait for message: "Waiting for 3 clients..."

**Step 2**: Double-click **`start_client0.bat`**

**Step 3**: Double-click **`start_client1.bat`**

**Step 4**: Double-click **`start_client2.bat`**

✅ Training will start automatically!

---

### Method 2: Terminal Commands

Open 4 separate terminals and run:

**Terminal 1 - Server:**
```cmd
cd c:\ai_disease_prediction
start_server.bat
```

**Terminal 2 - Client 0:**
```cmd
cd c:\ai_disease_prediction
start_client0.bat
```

**Terminal 3 - Client 1:**
```cmd
cd c:\ai_disease_prediction
start_client1.bat
```

**Terminal 4 - Client 2:**
```cmd
cd c:\ai_disease_prediction
start_client2.bat
```

---

## 📊 What You'll See

### Server Output:
```
Starting Flower server on localhost:8080
Waiting for 3 clients...

[Client connected: 0]
[Client connected: 1]  
[Client connected: 2]

============================================================
Round 1 - Training Results
============================================================
Client 0: Loss=0.6234, Acc=0.6500
Client 1: Loss=0.6189, Acc=0.6700
Client 2: Loss=0.6301, Acc=0.6400

Global Metrics: Loss=0.5985, Acc=0.6900
============================================================

Round 2 - Training Results
...
```

### Each Client Output:
```
[Client 0] Initialized with 140 training samples
[Client 0] Using device: cpu
[Client 0] Connecting to server at localhost:8080
[Client 0] Train Loss: 0.6234, Accuracy: 0.6500
[Client 0] Val Loss: 0.5987, Accuracy: 0.6800
```

---

## ⏱️ Training Time

- **10 rounds**: ~5-10 minutes
- **50 rounds**: ~20-30 minutes (better results)

The batch files are set to **10 rounds** by default for quick testing.

---

## 📈 After Training

Once training completes, you'll see:
```
✅ Best model saved: models/best_model.pt
```

Then:
1. **Refresh the dashboard** (press F5 in browser)
2. Dashboard will now show all features!
3. View model performance, feature importance, and make predictions

---

## 🎯 Expected Results

After 10 rounds:
- **Accuracy**: ~0.75-0.80
- **AUROC**: ~0.80-0.85
- **Model saved**: `models/best_model.pt`

---

## 💡 Pro Tips

1. **Start server first** - Always start server before clients
2. **Keep windows open** - Don't close terminals during training
3. **Watch server output** - Shows progress for all clients
4. **Be patient** - First round takes longer (model initialization)
5. **Dashboard stays open** - Keep it open, refresh after training

---

## 🔧 If Something Goes Wrong

### Clients can't connect to server:
- Make sure server is running first
- Check server shows "Waiting for 3 clients..."
- Verify firewall isn't blocking port 8080

### Training is too slow:
- This is normal for CPU training
- 10 rounds should complete in 5-10 minutes
- For faster testing, edit `start_server.bat` and change to `--rounds 5`

### Want to stop training:
- Press Ctrl+C in each terminal
- Or just close the terminal windows

---

## ✅ Ready to Start!

**Just double-click `start_server.bat` now!**

Then start the 3 clients and watch your federated learning system train!

---

**Questions?** Check [HOW_TO_RUN.md](HOW_TO_RUN.md) or [FIXED_AND_READY.md](FIXED_AND_READY.md)
