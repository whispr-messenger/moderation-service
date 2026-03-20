@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Lancement du script de telechargement d'images Google...
echo.
python -m tools.fetch_google_dataset %*
echo.
pause
