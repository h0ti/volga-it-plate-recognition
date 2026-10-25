@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON=python"
if exist "training\.venv\Scripts\python.exe" set "PYTHON=training\.venv\Scripts\python.exe"

echo === Volga-IT: synthetic detector training on the RTX computer ===
echo This command generates synthetic data and trains detector and OCR models.
echo It does not collect or automatically label real photos.
echo.

"%PYTHON%" -c "import sys; print('Python:', sys.version); import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo CUDA PyTorch is unavailable. Stop here; do not install packages blindly.
    pause
    exit /b 1
)
"%PYTHON%" -c "import ultralytics, yaml; print('Ultralytics:', ultralytics.__version__, 'PyYAML:', yaml.__version__)"
if errorlevel 1 (
    echo Ultralytics/PyYAML is not installed. Follow training\README.md before retrying.
    pause
    exit /b 1
)

nvidia-smi
if errorlevel 1 (
    echo nvidia-smi failed; stopping before data generation.
    pause
    exit /b 1
)
if exist "dataset\generated\meta.csv" (
    echo dataset\generated already exists. Move or rename it manually before retrying.
    pause
    exit /b 1
)
if exist "training\run_plates\prepared\data.yaml" (
    echo training\run_plates already has prepared data. Move or rename it manually.
    pause
    exit /b 1
)

start "RTX usage monitor - close after training" cmd /k nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 2

echo.
echo [1/3] Generating 6000 synthetic frames. Progress is printed every 500 frames.
"%PYTHON%" dataset\generator\build_dataset.py --out dataset\generated --synthetic 6000 --seed 20261024 --negative-fraction 0.10
if errorlevel 1 (
    echo Synthetic data generation failed. See the error above.
    pause
    exit /b 1
)

echo.
echo [2/3] Training detector. Ultralytics displays epoch/batch progress below.
echo Initial training may download yolov8n.pt; internet is needed only for setup/weights.
"%PYTHON%" training\train_detector.py --dataset dataset\generated --out training\run_plates --epochs 40 --imgsz 512 --batch 4 --workers 0 --device 0
if errorlevel 1 (
    echo Training failed. Keep logs and report the error; do not claim model completion.
    pause
    exit /b 1
)

echo.
echo [3/3] Training OCR on generated synthetic plates. Epoch progress is printed.
"%PYTHON%" training\train_ocr.py --out training\run_plates\ocr_best.pt --samples 12000 --epochs 20 --batch 128 --device 0
if errorlevel 1 (
    echo OCR training failed. Keep the detector weights and report the error.
    pause
    exit /b 1
)

echo.
echo Expected best model: training\run_plates\runs\plate_detector\weights\best.pt
echo OCR model: training\run_plates\ocr_best.pt
echo Copy weights to solution\weights only after checking licenses and validation results.
echo Both models are trained on synthetic data; real-image quality is unverified.
pause
