@echo off
REM Runs the (incremental) pipeline and launches the Streamlit demo UI.
REM Usage: run.bat [--reindex]
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo error: 'uv' is not installed. Run setup.bat first.
    exit /b 1
)

set REINDEX=0
if "%~1"=="--reindex" set REINDEX=1

echo === Ensuring Qdrant is running ===
docker compose up -d || exit /b 1

if not defined S3_ENDPOINT set S3_ENDPOINT=http://localhost:4566
curl -fs %S3_ENDPOINT% >nul 2>&1
if errorlevel 1 (
    echo error: S3/Floci not reachable at %S3_ENDPOINT%. Start Floci and re-run.
    exit /b 1
)

echo === Running chunk + embed + index pipeline ^(incremental^) ===
if %REINDEX%==1 (
    uv run python scripts\run_pipeline.py --reindex-all || exit /b 1
) else (
    uv run python scripts\run_pipeline.py || exit /b 1
)

echo === Launching Streamlit demo UI ===
uv run streamlit run app\ui\streamlit_app.py
