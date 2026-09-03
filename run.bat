@echo off
REM Runs the pipeline (if needed) and launches the Streamlit demo UI.
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

set NEED_PIPELINE=0
if %REINDEX%==1 set NEED_PIPELINE=1
if not exist data\clips\*.mp4 set NEED_PIPELINE=1

if %NEED_PIPELINE%==1 (
    dir /b data\raw_videos\*.mp4 data\raw_videos\*.mov data\raw_videos\*.mkv data\raw_videos\*.avi >nul 2>&1
    if errorlevel 1 (
        echo error: no source videos found in data\raw_videos\. Add some and re-run.
        exit /b 1
    )
    echo === Running chunk + embed + index pipeline ===
    uv run python scripts\run_pipeline.py || exit /b 1
) else (
    echo === data\clips\ already has clips -- skipping pipeline ^(use --reindex to force^) ===
)

echo === Launching Streamlit demo UI ===
uv run streamlit run app\ui\streamlit_app.py
