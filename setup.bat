@echo off
REM One-time environment setup: installs deps, creates .env, starts Qdrant.
REM Usage: setup.bat
setlocal enabledelayedexpansion
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo error: 'uv' is not installed. Install it from https://docs.astral.sh/uv/ and re-run.
    exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 (
    echo error: 'docker' is not installed ^(needed for Qdrant^). Install Docker and re-run.
    exit /b 1
)

echo === Installing Python dependencies ^(uv sync^) ===
uv sync || exit /b 1

if not exist .env (
    echo === Creating .env from .env.example ===
    copy .env.example .env >nul
    echo Fill in GROQ_API_KEY in .env if you want the Groq chat layer -- not required for plain search.
)

echo === Starting Qdrant ^(docker compose up -d^) ===
docker compose up -d || exit /b 1

if not defined QDRANT_HOST set QDRANT_HOST=localhost
if not defined QDRANT_PORT set QDRANT_PORT=6333

echo === Waiting for Qdrant to become healthy ===
set HEALTHY=0
for /l %%i in (1,1,30) do (
    if !HEALTHY! == 0 (
        curl -sf http://!QDRANT_HOST!:!QDRANT_PORT!/healthz >nul 2>&1
        if not errorlevel 1 (
            echo Qdrant is up.
            set HEALTHY=1
        ) else (
            timeout /t 1 /nobreak >nul
        )
    )
)
if !HEALTHY! == 0 (
    echo warning: Qdrant did not report healthy within 30s -- check 'docker compose logs'.
)

if not exist data\raw_videos mkdir data\raw_videos

echo.
echo Setup complete. Next steps:
echo   1. Place source videos in data\raw_videos\ ^(.mp4/.mov/.mkv/.avi^)
echo   2. Run run.bat
