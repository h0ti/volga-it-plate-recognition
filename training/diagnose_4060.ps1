$ErrorActionPreference = "Continue"

$reportPath = Join-Path $PSScriptRoot "gpu_report.txt"
$stageCount = 4

function Show-Stage {
    param(
        [int]$Number,
        [string]$Title
    )

    $percent = [math]::Floor(($Number / $stageCount) * 100)
    $filled = [math]::Floor($Number * 10 / $stageCount)
    $bar = ("#" * $filled) + ("-" * (10 - $filled))
    Write-Progress -Activity "Проверка RTX-компьютера" -Status "$percent% - $Title" -PercentComplete $percent
    Write-Host "[$bar] $percent% - $Title"
}

$lines = @(
    "Volga IT - GPU readiness report"
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')"
    "Windows version: $([Environment]::OSVersion.Version)"
    ""
    "== GPU and driver =="
)

Show-Stage -Number 1 -Title "Видеокарта и драйвер"
$nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    $lines += "nvidia-smi: NOT FOUND"
} else {
    $gpuInfo = & $nvidiaSmi.Source --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1
    $gpuExitCode = $LASTEXITCODE
    $lines += $gpuInfo | ForEach-Object { "$_" }
    if ($gpuExitCode -ne 0) {
        $lines += "nvidia-smi exit code: $gpuExitCode"
    }

    $lines += @("", "== Current GPU load (one read-only sample) ==")
    $gpuLoad = & $nvidiaSmi.Source --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>&1
    $loadExitCode = $LASTEXITCODE
    $lines += $gpuLoad | ForEach-Object { "$_" }
    if ($loadExitCode -ne 0) {
        $lines += "GPU load query exit code: $loadExitCode"
    }
}

$lines += @("", "== Python and PyTorch (read-only check) ==")
Show-Stage -Number 2 -Title "Python и PyTorch CUDA"
$pythonLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
$pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
$pythonProbe = @'
import importlib.util
import sys
print("Python:", sys.version.split()[0])
if importlib.util.find_spec("torch") is None:
    print("PyTorch: NOT INSTALLED")
else:
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("PyTorch CUDA build:", torch.version.cuda or "CPU-only")
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
            print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    except Exception as exc:
        print("PyTorch import error:", type(exc).__name__, str(exc)[:300])
'@

if ($null -ne $pythonLauncher) {
    $probeOutput = $pythonProbe | & $pythonLauncher.Source -3 - 2>&1
    $probeExitCode = $LASTEXITCODE
    $lines += $probeOutput | ForEach-Object { "$_" }
    if ($probeExitCode -ne 0) {
        $lines += "Python probe exit code: $probeExitCode"
    }
} elseif ($null -ne $pythonCommand) {
    $probeOutput = $pythonProbe | & $pythonCommand.Source - 2>&1
    $probeExitCode = $LASTEXITCODE
    $lines += $probeOutput | ForEach-Object { "$_" }
    if ($probeExitCode -ne 0) {
        $lines += "Python probe exit code: $probeExitCode"
    }
} else {
    $lines += "Python: NOT FOUND (no installation was attempted)"
}

$lines += @("", "== Processor and available disk space ==")
$processorInfo = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty Name
$processorLoad = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty LoadPercentage
if ($processorInfo) {
    $lines += "CPU: $processorInfo"
} else {
    $lines += "CPU: unavailable"
}
if ($null -ne $processorLoad) {
    $lines += "Current CPU load (approx.): $processorLoad%"
}

$osMemory = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
if ($null -ne $osMemory -and $osMemory.TotalVisibleMemorySize -gt 0) {
    $totalRamGiB = [math]::Round($osMemory.TotalVisibleMemorySize / 1MB, 1)
    $freeRamGiB = [math]::Round($osMemory.FreePhysicalMemory / 1MB, 1)
    $usedRamGiB = [math]::Round($totalRamGiB - $freeRamGiB, 1)
    $lines += "Current RAM (approx.): $usedRamGiB / $totalRamGiB GiB used"
}

$driveLetter = [System.IO.Path]::GetPathRoot($PSScriptRoot).Substring(0, 1)
$drive = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
if ($null -ne $drive) {
    $freeGiB = [math]::Round($drive.Free / 1GB, 1)
    $lines += "Free space on project drive ($driveLetter`:): $freeGiB GiB"
} else {
    $lines += "Free space: unavailable"
}

Show-Stage -Number 3 -Title "Нагрузка CPU/RAM и свободное место"
$lines += @(
    ""
    "This report does not contain API keys, does not install software,"
    "does not download data, and does not run training."
)

Show-Stage -Number 4 -Title "Сохранение отчёта"
Set-Content -LiteralPath $reportPath -Value $lines -Encoding UTF8
Write-Progress -Activity "Проверка RTX-компьютера" -Completed
Write-Output "Diagnostic complete."
Write-Output "Report: $reportPath"
