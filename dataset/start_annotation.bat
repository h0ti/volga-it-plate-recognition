@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=python"
if exist "training\.venv\Scripts\python.exe" set "PYTHON=training\.venv\Scripts\python.exe"

"%PYTHON%" -c "import tkinter; from PIL import Image" >nul 2>nul
if errorlevel 1 (
    echo Python/Tkinter or Pillow is missing.
    echo Use Python 3.10+ with Tkinter and install Pillow with:
    echo python -m pip install Pillow
    echo If training\.venv exists, run:
    echo training\.venv\Scripts\python.exe -m pip install Pillow
    pause
    exit /b 1
)

echo Starting the local annotation tool. Photos and labels stay on this PC.
echo If the window closes unexpectedly, the error log is:
echo dataset\real_review_inbox\annotation_errors.log
"%PYTHON%" dataset\annotate_images.py
if errorlevel 1 (
    echo Annotation tool stopped with an error. Check:
    echo dataset\real_review_inbox\annotation_errors.log
    pause
    exit /b 1
)
