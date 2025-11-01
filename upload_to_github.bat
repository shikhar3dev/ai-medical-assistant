@echo off
echo ============================================================
echo 🚀 UPLOADING AI MEDICAL ASSISTANT TO GITHUB
echo ============================================================
echo.
echo 👤 GitHub Username: shikhar3dev
echo 📁 Repository: ai-medical-assistant
echo 🏥 Project: Privacy-Preserving Federated Learning for Healthcare AI
echo.

cd /d "%~dp0"

echo [1/6] Initializing Git repository...
git init

echo.
echo [2/6] Adding all files...
git add .

echo.
echo [3/6] Creating initial commit...
git commit -m "🏥 AI Medical Assistant - Privacy-Preserving Federated Learning for Healthcare"

echo.
echo [4/6] Adding GitHub remote...
git remote add origin https://github.com/shikhar3dev/ai-medical-assistant.git

echo.
echo [5/6] Setting main branch...
git branch -M main

echo.
echo [6/6] Pushing to GitHub...
git push -u origin main

echo.
echo ============================================================
echo ✅ UPLOAD COMPLETE!
echo ============================================================
echo.
echo 🌐 Your repository is now live at:
echo https://github.com/shikhar3dev/ai-medical-assistant
echo.
echo 🚀 Next steps:
echo 1. Go to https://share.streamlit.io
echo 2. Sign in with GitHub
echo 3. Click "New app"
echo 4. Select: shikhar3dev/ai-medical-assistant
echo 5. Main file: enhanced_dashboard.py
echo 6. Click "Deploy"
echo.
echo 🎉 Your AI Medical Assistant will be live on the internet!
echo ===========================================================

pause
