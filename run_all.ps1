<#
.SYNOPSIS
  End-to-end Acoustic-JEPA pipeline: dataset -> train -> probe -> plan.
  GPU-friendly defaults (large batch, AMP; device auto-detected by the scripts).

.DESCRIPTION
  Runs the four jepa/*.py stages in order against a dataset produced by
  jepa_dump.exe. See ACOUSTIC_JEPA_SPEC.md. Stages can be skipped to iterate.

  Provisioning the torch venv is optional (-SetupVenv, needs `uv` on PATH). On a
  CUDA box install the CUDA build via -TorchIndex; the Python scripts then pick
  cuda automatically. AMP is always passed and is a no-op on CPU.

.EXAMPLE
  # first time on a CUDA box: make the venv, then run the full pipeline
  ./run_all.ps1 -SetupVenv -Episodes 2000 -Epochs 60 -Batch 256

.EXAMPLE
  # reuse an existing dataset + venv, just retrain and evaluate
  ./run_all.ps1 -SkipData -Epochs 80 -Batch 512

.EXAMPLE
  # CPU smoke test of the whole chain
  ./run_all.ps1 -SetupVenv -TorchIndex https://download.pytorch.org/whl/cpu `
                -Episodes 40 -Steps 24 -Epochs 10 -Batch 64 -Device cpu
#>
[CmdletBinding()]
param(
    [int]    $Episodes   = 2000,
    [int]    $Steps      = 32,
    [int]    $Horizon    = 8,
    [int]    $Window     = 4,
    [int]    $Batch      = 256,
    [int]    $Epochs     = 60,
    [double] $Lr         = 1e-3,
    [double] $VarCoeff   = 4.0,
    [double] $CovCoeff   = 1.0,
    [int]    $DumpThreads = 8,
    [int]    $NumWorkers = 4,
    [int]    $Seed       = 1,
    [string] $Device     = "",                          # "" => let the scripts auto-detect (cuda if available)
    [string] $DataDir    = "$PSScriptRoot\results\jepa_data",
    [string] $RunDir     = "$PSScriptRoot\results\runs\jepa",
    [string] $Venv       = "$PSScriptRoot\.venv",
    [switch] $SetupVenv,
    [string] $TorchIndex = "https://download.pytorch.org/whl/cu124",
    [switch] $SkipData,
    [switch] $SkipTrain,
    [switch] $SkipProbe,
    [switch] $SkipPlan
)

$ErrorActionPreference = "Continue"    # we check $LASTEXITCODE ourselves
$SteamDir = $PSScriptRoot
$JepaDir  = Join-Path $SteamDir "jepa"
$Sdk      = Join-Path $SteamDir "..\Plugins\SteamAudio\Source\SteamAudioSDK"
$Py       = Join-Path $Venv "Scripts\python.exe"

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg)    { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
function Check($stage) { if ($LASTEXITCODE -ne 0) { Fail "$stage (exit $LASTEXITCODE)" } }
function Step([scriptblock]$b, $name) {
    $t0 = Get-Date                    # NB: not Measure-Command -- that swallows the block's stdout
    & $b
    Write-Host ("  [{0}] {1:n1}s" -f $name, ((Get-Date) - $t0).TotalSeconds) -ForegroundColor DarkGray
}

# --- 0. venv (optional) -----------------------------------------------------
if ($SetupVenv) {
    Section "Provisioning torch venv ($Venv)"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail "uv not found. Install it first:  pip install uv   (then re-run with -SetupVenv)"
    }
    uv venv --python 3.12 $Venv;                                   Check "uv venv"
    uv pip install --python $Py --index-url $TorchIndex torch;     Check "uv pip install torch"
    uv pip install --python $Py numpy;                             Check "uv pip install numpy"
}
if (-not (Test-Path $Py)) {
    Fail "no venv python at $Py. Run with -SetupVenv, or point -Venv at an existing torch venv."
}
Section "Environment"
& $Py -c "import torch,numpy; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| numpy',numpy.__version__)"
Check "torch import"

# device flag passed to the python scripts ("" => their own auto-detect)
$DeviceArg = @(); if ($Device -ne "") { $DeviceArg = @("--device", $Device) }

# --- 1. dataset -------------------------------------------------------------
$Dump = Join-Path $SteamDir "tools\jepa_dump.exe"
if (-not $SkipData) {
    Section "Dataset ($Episodes episodes x $Steps steps -> $DataDir)"
    if (-not (Test-Path $Dump)) {
        Write-Host "  jepa_dump.exe missing -> building via src\build.bat" -ForegroundColor Yellow
        Push-Location (Join-Path $SteamDir "src")
        cmd /c build.bat
        Pop-Location
        Check "build.bat"
    }
    $dll = Join-Path $SteamDir "tools\phonon.dll"
    if (-not (Test-Path $dll)) { Copy-Item (Join-Path $Sdk "lib\windows-x64\phonon.dll") (Join-Path $SteamDir "tools") -Force }
    Step { & $Dump $Episodes $DataDir $Steps $Seed $DumpThreads } "dump"
    Check "jepa_dump"
} else {
    Section "Dataset (reusing $DataDir)"
}
if (-not (Test-Path $DataDir)) { Fail "no dataset at $DataDir" }
$nEp = (Get-ChildItem "$DataDir\*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "  episodes on disk: $nEp"

Push-Location $JepaDir
try {
    # --- 2. train -----------------------------------------------------------
    if (-not $SkipTrain) {
        Section "Train ($Epochs epochs, batch $Batch, horizon $Horizon)"
        Step {
            & $Py train.py $DataDir --out $RunDir --epochs $Epochs --batch $Batch `
                --horizon $Horizon --window $Window --lr $Lr --var-coeff $VarCoeff --cov-coeff $CovCoeff `
                --num-workers $NumWorkers --amp --seed $Seed @DeviceArg
        } "train"
        Check "train.py"
    } else { Section "Train (skipped)" }

    $Ckpt = Join-Path $RunDir "final.pt"
    if (-not (Test-Path $Ckpt)) { $Ckpt = Join-Path $RunDir "latest.pt" }

    # --- 3. probe -----------------------------------------------------------
    if (-not $SkipProbe) {
        Section "Probe (frozen-latent geometry, spec 4a)"
        if (-not (Test-Path $Ckpt)) { Fail "no checkpoint in $RunDir" }
        Step { & $Py probe.py $Ckpt $DataDir --num-workers $NumWorkers @DeviceArg } "probe"
        Check "probe.py"
    } else { Section "Probe (skipped)" }

    # --- 4. plan ------------------------------------------------------------
    if (-not $SkipPlan) {
        Section "Plan (world-model rollout + latent MPC, spec 4b)"
        if (-not (Test-Path $Ckpt)) { Fail "no checkpoint in $RunDir" }
        Step { & $Py plan.py $Ckpt $DataDir --horizon $Horizon --num-workers $NumWorkers @DeviceArg } "plan"
        Check "plan.py"
    } else { Section "Plan (skipped)" }
}
finally { Pop-Location }

Section "Done"
Write-Host "  checkpoints: $RunDir"
Write-Host "  results:     $RunDir\probe_results.json , $RunDir\plan_results.json , $RunDir\log.csv"
