$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "."
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$logPath = "logs\fullrag80_detector.log"
$transcriptPath = "logs\fullrag80_detector.transcript.txt"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
Start-Transcript -Path $transcriptPath -Force | Out-Null

Write-Host "Starting fullrag80_detector"
Write-Host "Progress format: [progress done/total percent elapsed eta] id detector action abstain expected_behavior include_risk complete"
Write-Host "Log file: $logPath"
Write-Host "Transcript: $transcriptPath"
Write-Host ""

& ".\.venv\Scripts\python.exe" -u "scripts\run_finreg_real_life_benchmark.py" `
  --mode full-rag `
  --config "gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_section_quality_local" `
  --k 8 `
  --run-name "fullrag80_detector" 2>&1 | Tee-Object -FilePath $logPath

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "fullrag80_detector finished with exit code $exitCode"
Stop-Transcript | Out-Null
Write-Host "Press Enter to close this window."
Read-Host | Out-Null
exit $exitCode
