#!/bin/bash

# GNN Trade Forecasting System - Comprehensive Startup Script

# 1. Setup environment
PROJECT_ROOT=$(pwd)
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"
export PYTHONPATH=$PROJECT_ROOT

echo "🚀 Starting GNN Trade Forecasting Full Project..."

# 2. Kill existing processes
echo "🧹 Cleaning up old processes (API on port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# 3. Data Pipeline
echo "\n📊 Step 1/3: Running Data Preprocessing..."
$VENV_PYTHON scripts/preprocess_data.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed. Check logs/preprocessing.log"
    exit 1
fi

# Check if model already exists
if [ ! -f "models/gnn_working.pt" ]; then
    echo "\n🧠 Step 2/3: Training Model (new)..."
    $VENV_PYTHON scripts/train_model.py
else
    echo "\n🧠 Step 2/3: Using existing model (gnn_working.pt)..."
fi

# Ensure the latest model is copied to gnn_working.pt if needed (optional)
latest_model=$(ls -t models/gnn_*.pt 2>/dev/null | head -1)
if [ ! -z "$latest_model" ]; then
    cp "$latest_model" models/gnn_working.pt
fi

# 4. Start Services
echo "\n🌐 Step 3/3: Starting Services (API & Dashboard)..."

# Start FastAPI Backend in background
echo "📡 Starting FastAPI Backend on http://localhost:8000..."
nohup $VENV_PYTHON src/api/main.py > logs/api.log 2>&1 &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API to initialize..."
sleep 5

# Check if API is running
if ps -p $API_PID > /dev/null; then
    echo "✅ API is running (PID: $API_PID). Logs at logs/api.log"
else
    echo "❌ API failed to start. Check logs/api.log"
    exit 1
fi

# Start Next.js Frontend
echo "🖥️  Starting Next.js Dashboard..."
cd dashboard/src && npm run dev
