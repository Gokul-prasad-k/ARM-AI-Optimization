    # ============================================
# Early Exit DS-CNN Validation Pipeline
# ============================================

Write-Host ""
Write-Host "========================================="
Write-Host " Early Exit DS-CNN Validation Pipeline"
Write-Host "========================================="
Write-Host ""
# Repository paths
Write-Host ""
Write-Host "========================================="
Write-Host " Early Exit DS-CNN Validation Pipeline"
Write-Host "========================================="
Write-Host ""

$ROOT = $PSScriptRoot
$REPO_ROOT = (Resolve-Path "$ROOT\..\..").Path

# Required by MLCommons on Windows
$env:HOME = $REPO_ROOT
$env:PWD  = $ROOT

Set-Location $ROOT

Write-Host "[1/5] Splitting original model..."
python split_model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/5] Validating split..."
python validate_split.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/5] Running final validation..."
python final_validation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/5] Running compute analysis..."
python compute_analysis.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/5] Building deployment handoff..."
python build_handoff.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========================================="
Write-Host " VALIDATION COMPLETE"
Write-Host "========================================="
Write-Host ""
Write-Host "Generated artifacts:"
Write-Host " - results/ee1_threshold_results.csv"
Write-Host " - FINAL_HANDOFF/"