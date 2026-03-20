@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Mode dry-run (aucun telechargement)...
echo.
python -m tools.fetch_google_dataset --dry-run
echo.
pause
