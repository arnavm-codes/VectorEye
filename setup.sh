#!/usr/bin/env bash
# One-time environment setup: installs deps, creates .env, starts Qdrant.
# Usage: ./setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is not installed. Install it from https://docs.astral.sh/uv/ and re-run." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "error: 'docker' is not installed (needed for Qdrant). Install Docker and re-run." >&2
    exit 1
fi

echo "=== Installing Python dependencies (uv sync) ==="
uv sync

if [ ! -f .env ]; then
    echo "=== Creating .env from .env.example ==="
    cp .env.example .env
    echo "Fill in GROQ_API_KEY in .env if you want the Groq chat layer -- not required for plain search."
fi

echo "=== Starting Qdrant (docker compose up -d) ==="
docker compose up -d

echo "=== Waiting for Qdrant to become healthy ==="
for i in $(seq 1 30); do
    if curl -sf http://"${QDRANT_HOST:-localhost}":"${QDRANT_PORT:-6333}"/healthz >/dev/null 2>&1; then
        echo "Qdrant is up."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "warning: Qdrant did not report healthy within 30s -- check 'docker compose logs'." >&2
    fi
    sleep 1
done

mkdir -p data/raw_videos

echo
echo "Setup complete. Next steps:"
echo "  1. Place source videos in data/raw_videos/ (.mp4/.mov/.mkv/.avi)"
echo "  2. Run ./run.sh"
