@echo off
setlocal
cd /d "%~dp0\.."

echo === Preparing isolated Python environment for RTX 4060 ===
echo Requires Python 3.12 x64 already installed and internet access.
echo This installs PyTorch CUDA 12.8 and Ultralytics into training\.venv.
echo It does not install a system CUDA toolkit or require administrator rights.
echo.

py -3.12 --version
if errorlevel 1 (
    echo Python 3.12 was not found. Install Python 3.12 x64 from python.org first.
    pause
    exit /b 1
)

if not exist "training\.venv\Scripts\python.exe" (
    py -3.12 -m venv training\.venv
    if errorlevel 1 (
        echo Could not create the local virtual environment.
        pause
        exit /b 1
    )
)

training\.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo pip upgrade failed.
    pause
    exit /b 1
)

echo [1/2] Installing official PyTorch CUDA 12.8 wheels.
training\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo PyTorch installation failed. Save the error and do not continue to training.
    pause
    exit /b 1
)

echo [2/2] Installing project training requirements.
training\.venv\Scripts\python.exe -m pip install -r training\requirements.txt
if errorlevel 1 (
    echo Training dependencies failed to install.
    pause
    exit /b 1
)

training\.venv\Scripts\python.exe -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo CUDA verification failed. Do not run training_semifinal.bat.
    pause
    exit /b 1
)

echo.
echo Setup complete. Next, check the Ultralytics/weights license and run:
echo training\train_semifinal.bat
pause
