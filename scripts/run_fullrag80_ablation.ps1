$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "."
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$runs = @(
    @{
        Name = "fullrag80_baseline"
        Config = "gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_section_quality_local"
        Extra = @("--disable-detector", "--disable-gating", "--disable-answer-quality")
    },
    @{
        Name = "fullrag80_detector_only"
        Config = "gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_section_quality_local"
        Extra = @("--disable-answer-quality")
    },
    @{
        Name = "fullrag80_stochastic_only"
        Config = "gating_finreg_local_qwen3_ablation_stochastic"
        Extra = @("--disable-answer-quality")
    },
    @{
        Name = "fullrag80_detector_stochastic"
        Config = "gating_finreg_local_qwen3_ablation_detector_stochastic"
        Extra = @("--disable-answer-quality")
    }
)

foreach ($run in $runs) {
    Write-Host ""
    Write-Host "==== Running $($run.Name) ===="
    Write-Host "Config: $($run.Config)"
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total --format=csv

    $cmd = @(
        "-u",
        "scripts\run_finreg_real_life_benchmark.py",
        "--mode", "full-rag",
        "--config", $run.Config,
        "--k", "8",
        "--run-name", $run.Name
    ) + $run.Extra

    & ".\.venv\Scripts\python.exe" $cmd 2>&1 | Tee-Object -FilePath "logs\$($run.Name).log"

    Write-Host "==== Completed $($run.Name) ===="
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total --format=csv
}

Write-Host ""
Write-Host "All ablation runs finished. Press Enter to close this window."
Read-Host
