# PowerShell script to set up virtual environment and install project dependencies

$ErrorActionPreference = "Stop"

Write-Host "--------------------------------------------------" -ForegroundColor Cyan
Write-Host "   Setting up Indian AQI Project Environment   " -ForegroundColor Cyan
Write-Host "--------------------------------------------------" -ForegroundColor Cyan

# Create venv if it does not exist
if (-not (Test-Path -Path "aqi_env")) {
    Write-Host "[INFO] Creating virtual environment 'aqi_env'..." -ForegroundColor Yellow
    & py -m venv aqi_env
    Write-Host "[SUCCESS] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[INFO] Virtual environment 'aqi_env' already exists." -ForegroundColor Yellow
}

# Upgrade pip and install requirements
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Yellow
& .\aqi_env\Scripts\python.exe -m pip install --upgrade pip

Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .\aqi_env\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "[SUCCESS] Setup complete! To activate this environment, run:" -ForegroundColor Green
Write-Host ".\aqi_env\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "--------------------------------------------------" -ForegroundColor Cyan
