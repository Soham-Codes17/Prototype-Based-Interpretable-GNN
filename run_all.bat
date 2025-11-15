@echo off
title Prototype-Based Interpretable GNN Runner
echo ==========================================
echo   Starting GNN Training Pipeline
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

echo [1/3] Training Baseline GNN...
python train_gnns.py datasets.dataset_name=mutag
if %errorlevel% neq 0 (
    echo Error: Baseline GNN training failed.
    pause
    exit /b
)

echo [2/3] Training ProtoPNet with Pretrained GNN...
python train_protopgnns.py datasets.dataset_name=mutag use_pretrained=true
if %errorlevel% neq 0 (
    echo Error: ProtoPNet training failed.
    pause
    exit /b
)

echo [3/3] Training TesNet with Pretrained GNN...
python train_tesgnns.py datasets.dataset_name=mutag use_pretrained=true
if %errorlevel% neq 0 (
    echo Error: TesNet training failed.
    pause
    exit /b
)

echo.
echo ==========================================
echo   ✅ All trainings completed successfully!
echo ==========================================
pause
