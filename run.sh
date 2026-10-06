#!/usr/bin/env bash
# Runs the (incremental) pipeline and launches the Streamlit demo UI.
# Usage: ./run.sh [--reindex]
#   --reindex  re-embed clips even if they are already indexed.
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

S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:4566}"
if ! curl -fs "$S3_ENDPOINT" >/dev/null 2>&1; then
    echo "error: S3/Floci not reachable at $S3_ENDPOINT. Start Floci and re-run." >&2
    exit 1
fi

echo "=== Running chunk + embed + index pipeline (incremental) ==="
if [ "$REINDEX" -eq 1 ]; then
    uv run python scripts/run_pipeline.py --reindex-all
else
    uv run python scripts/run_pipeline.py
fi

echo "=== Launching Streamlit demo UI ==="
exec uv run streamlit run app/ui/streamlit_app.py
