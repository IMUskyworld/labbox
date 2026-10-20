# Local quality gate: run this before pushing.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1 [-NoGpu]
# （本机默认执行策略会拦截 -File；-ExecutionPolicy Bypass 是必须的——faillab 事故 F-0001）

param(
    [switch]$NoGpu = $true
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $root ".venv/bin/python"
}
if (-not (Test-Path $py)) {
    Write-Host "no .venv found; run:  uv venv && uv pip install -e '.[dev]'" -ForegroundColor Red
    exit 2
}

function Invoke-Step($name, $argv) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    & $py @argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $name" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Step "ruff check" @("-m", "ruff", "check", "src", "tests", "scripts", "examples")
Invoke-Step "ruff format --check" @("-m", "ruff", "format", "--check", "src")
Invoke-Step "pyright (src + examples)" @("-m", "pyright")
Invoke-Step "schemas are current" @("scripts/export_schemas.py")

if ($NoGpu) {
    Invoke-Step "pytest (no gpu)" @("-m", "pytest", "-q", "-m", "not gpu")
} else {
    Invoke-Step "pytest (all)" @("-m", "pytest", "-q")
}

Invoke-Step "MVP acceptance audit (DEVELOPMENT.md 21)" @("scripts/verify_mvp.py")

Write-Host "`nall checks passed" -ForegroundColor Green
