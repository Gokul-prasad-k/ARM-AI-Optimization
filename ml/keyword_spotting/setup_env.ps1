python -m venv .venv

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r .\ml\keyword_spotting\requirements-lock.txt

if (!(Test-Path ".\mlcommons_data")) {
    Write-Host "Downloading MLCommons Speech Commands dataset..."
    Push-Location .\ml\keyword_spotting
    python train.py --data_dir ../../mlcommons_data
    Pop-Location
}
else {
    Write-Host "Dataset already exists. Skipping download."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate later using:"
Write-Host ".\.venv\Scripts\Activate.ps1"   