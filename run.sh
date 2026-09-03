#!/usr/bin/env bash
# Runs the pipeline (if needed) and launches the Streamlit demo UI.
# Usage: ./run.sh [--reindex]
#   --reindex  force a fresh chunk+embed+index pass even if data/clips/
#              already has clips (e.g. after adding new source videos).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REINDEX=0
if [ "${1:-}" = "--reindex" ]; then
    REINDEX=1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is not installed. Run ./setup.sh first." >&2
    exit 1
fi

echo "=== Ensuring Qdrant is running ==="
docker compose up -d

if [ "$REINDEX" -eq 1 ] || [ -z "$(find data/clips -maxdepth 1 -name '*.mp4' -print -quit 2>/dev/null)" ]; then
    if [ -z "$(find data/raw_videos -maxdepth 1 \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' \) -print -quit 2>/dev/null)" ]; then
        echo "error: no source videos found in data/raw_videos/. Add some and re-run." >&2
        exit 1
    fi
    echo "=== Running chunk + embed + index pipeline ==="
    uv run python scripts/run_pipeline.py
else
    echo "=== data/clips/ already has clips -- skipping pipeline (use --reindex to force) ==="
fi

echo "=== Launching Streamlit demo UI ==="
exec uv run streamlit run app/ui/streamlit_app.py
