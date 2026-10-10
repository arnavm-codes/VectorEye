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

REM Clip URLs are HMAC-signed with this; the API refuses to start without it.
findstr /b /r /c:"CLIP_SIGNING_SECRET=." .env >nul 2>&1
if errorlevel 1 (
    echo === Generating CLIP_SIGNING_SECRET in .env ===
    uv run python -c "import re,secrets,pathlib; p=pathlib.Path('.env'); s=p.read_text(); k='CLIP_SIGNING_SECRET='+secrets.token_hex(32); p.write_text(re.sub(r'(?m)^CLIP_SIGNING_SECRET=.*$',k,s) if re.search(r'(?m)^CLIP_SIGNING_SECRET=',s) else s.rstrip()+'\n'+k+'\n')" || exit /b 1
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

if not defined S3_ENDPOINT set S3_ENDPOINT=http://localhost:4566
curl -fs !S3_ENDPOINT! >nul 2>&1
if errorlevel 1 (
    echo warning: S3/Floci not reachable at !S3_ENDPOINT! -- start it, then run "uv run python scripts/ensure_buckets.py" to create the buckets.
) else (
    echo === Ensuring S3 buckets exist ===
    uv run python scripts/ensure_buckets.py || exit /b 1
)

REM Download every model's weights now (Long-CLIP, OpenAI CLIP, Whisper, the transcript
REM sentence model, and the YOLO-World detector), regardless of which optional
REM features are enabled in .env, so nothing downloads mid-pipeline or on the
REM first search. Cached files are skipped, so re-running setup is cheap.
echo === Downloading all model weights ^(one-time, ~1.3 GB^) ===
uv run python scripts\download_models.py || exit /b 1

echo.
echo Setup complete. Next steps:
echo   1. Upload source videos to the raw-videos bucket on your S3/Floci endpoint
echo   2. Run run.bat
