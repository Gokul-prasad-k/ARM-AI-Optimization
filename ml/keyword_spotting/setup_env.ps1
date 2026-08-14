python -m venv .venv

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r .\requirements-lock.txt

if (!(Test-Path "..\..\mlcommons_data")) {
    Write-Host "Downloading MLCommons Speech Commands dataset..."
    $env:HOME = (Resolve-Path "..\..\").Path
    $env:PWD  = (Get-Location).Path
    python train.py --data_dir ..\..\mlcommons_data
}
else {
    Write-Host "Dataset already exists. Skipping download."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate later using:"
Write-Host ".\.venv\Scripts\Activate.ps1"