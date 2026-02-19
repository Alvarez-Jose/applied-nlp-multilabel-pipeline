# activate_ml.ps1
Write-Host "Activating ML/NLP Environment..." -ForegroundColor Green
& "C:\Users\Antonio Alvarez\ml_nlp_env\Scripts\Activate.ps1"
Write-Host "✅ Environment activated!" -ForegroundColor Green
Write-Host "Python: $(python --version)" -ForegroundColor Cyan
Write-Host "Current Directory: $(Get-Location)" -ForegroundColor Yellow