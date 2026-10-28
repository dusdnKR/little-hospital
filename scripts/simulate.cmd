@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m clinical_memory simulate %*
) else (
  python -m clinical_memory simulate %*
)
