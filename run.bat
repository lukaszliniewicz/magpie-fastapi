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

:: Check if environment is ready
if not exist ".pixi\envs\default\conda-meta" (
    echo [SETUP] Installing pixi environment first run - this may take a while...
    pixi install
    if %ERRORLEVEL% neq 0 (
        echo ERROR: pixi install failed
        pause
        exit /b 1
    )
) else (
    echo [OK] Environment ready
)

set "MAGPIE_PORT=8030"
set "MAGPIE_HOST=0.0.0.0"

echo.
echo Starting Magpie TTS API server on port %MAGPIE_PORT%...
echo Device: %DEVICE%
echo Voice catalog: check /v1/audio/voices
echo.
echo Server will start in a few minutes while the model downloads and loads...
echo.

pixi run python -m uvicorn main:app --host %MAGPIE_HOST% --port %MAGPIE_PORT% --log-level info

if %ERRORLEVEL% neq 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%
    pause
)

popd
