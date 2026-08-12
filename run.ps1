$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  python -m venv .venv
  .\.venv\Scripts\pip install -r requirements.txt
}
Write-Host "Federated Health Demo -> http://127.0.0.1:8099" -ForegroundColor Green
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8099 --reload
