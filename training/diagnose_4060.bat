@echo off
setlocal
cd /d "%~dp0"

echo Checking this computer without installing packages or training...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0diagnose_4060.ps1"
if errorlevel 1 (
    echo.
    echo The diagnostic did not finish. Send the error text shown above.
    pause
    exit /b 1
)

echo.
echo Report saved to: "%~dp0gpu_report.txt"
echo Send gpu_report.txt back. Do not send passwords or API keys.
pause
exit /b 0
