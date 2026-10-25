@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON=python"
if exist "training\.venv\Scripts\python.exe" set "PYTHON=training\.venv\Scripts\python.exe"

echo === Resuming detector from the last saved epoch ===
echo Reuses dataset\generated and training\run_plates. It does not regenerate/copy 6 GB.
echo DataLoader workers=0, batch=4, image-size=512 to reduce Windows RAM pressure.
echo.

if not exist "dataset\generated\meta.csv" (
    echo Missing dataset\generated\meta.csv. Keep the original generated dataset in place.
    pause
    exit /b 1
)
if not exist "training\run_plates\prepared\data.yaml" (
    echo Missing prepared YOLO data. Do not delete the old training\run_plates folder.
    pause
    exit /b 1
)
if not exist "training\run_plates\runs\plate_detector\weights\last.pt" (
    echo Missing last.pt checkpoint. This run cannot resume; keep the logs and report this.
    pause
    exit /b 1
)

"%PYTHON%" -c "import torch; print('CUDA:', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo CUDA is unavailable; no training was started.
    pause
    exit /b 1
)

start "RTX usage monitor - close after training" cmd /k nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 2
"%PYTHON%" training\train_detector.py --dataset dataset\generated --out training\run_plates --resume --imgsz 512 --batch 4 --workers 0 --device 0
if errorlevel 1 (
    echo Resume stopped. Keep best.pt, last.pt, results.csv and this error log.
    pause
    exit /b 1
)

echo.
echo Detector resume finished. Keep training\run_plates and its runs folder.
pause
