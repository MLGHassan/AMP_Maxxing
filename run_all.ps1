# Runs the complete DeepAMP pipeline on Windows (PowerShell).
#
#   .\run_all.ps1                 full run
#   .\run_all.ps1 -Quick          ~15 min reduced run, to check the install
#   .\run_all.ps1 -Force          redo every step
#
# This is a thin wrapper: all the logic lives in run_all.py, so Windows, macOS and
# Linux run exactly the same steps.
param(
    [switch]$Quick,
    [switch]$Force,
    [int]$From = 1,
    [switch]$List
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Prefer the virtual environment in .venv if the user made one
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Error "No 'python' on PATH. Install Python 3.10+ from python.org and tick 'Add python.exe to PATH'."
    }
    $py = $cmd.Source
}
Write-Host "Using interpreter: $py"

$argv = @("run_all.py", "--from", $From)
if ($Quick) { $argv += "--quick" }
if ($Force) { $argv += "--force" }
if ($List)  { $argv += "--list" }

& $py @argv
exit $LASTEXITCODE
