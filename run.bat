@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

echo ============================================================
echo  Magpie TTS - Pandrator Backend
echo  (NVIDIA Magpie TTS Multilingual 357M via NeMo Framework)
echo ============================================================
echo.

:: Check if pixi is available
where pixi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: pixi is not installed. Install it from https://pixi.sh
    pause
    exit /b 1
)

:: Determine if CUDA GPU is available (check via nvidia-smi)
set "DEVICE=cpu"
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] NVIDIA GPU detected
    set "DEVICE=cuda"
) else (
    echo [INFO] No NVIDIA GPU detected, using CPU (will be very slow)
)

:: Allow override via environment variable MAGPIE_DEVICE
if not "%MAGPIE_DEVICE%"=="" (
    set "DEVICE=%MAGPIE_DEVICE%"
)

set "MAGPIE_PORT=8030"
set "MAGPIE_HOST=0.0.0.0"

pixi run python run.py --host %MAGPIE_HOST% --port %MAGPIE_PORT% --device %DEVICE%

if %ERRORLEVEL% neq 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%
    pause
)

popd
