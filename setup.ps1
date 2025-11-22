Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Host "`nSetup complete! 🎉"
Write-Host "Now place your service_account.json file in the project root."
