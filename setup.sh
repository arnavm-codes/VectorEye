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

# Clip URLs are HMAC-signed with this; the API refuses to start without it.
if ! grep -qE '^CLIP_SIGNING_SECRET=.+' .env; then
    echo "=== Generating CLIP_SIGNING_SECRET in .env ==="
    SECRET="$(uv run python -c 'import secrets; print(secrets.token_hex(32))')"
    if grep -qE '^CLIP_SIGNING_SECRET=' .env; then
        sed -i "s/^CLIP_SIGNING_SECRET=.*/CLIP_SIGNING_SECRET=$SECRET/" .env
    else
        printf '\nCLIP_SIGNING_SECRET=%s\n' "$SECRET" >> .env
    fi
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

S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:4566}"
if ! curl -fs "$S3_ENDPOINT" >/dev/null 2>&1; then
    echo "warning: S3/Floci not reachable at $S3_ENDPOINT -- start it, then run 'uv run python scripts/ensure_buckets.py' to create the buckets." >&2
else
    echo "=== Ensuring S3 buckets exist ==="
    uv run python scripts/ensure_buckets.py
fi

# Download every model's weights now (Long-CLIP, OpenAI CLIP, Whisper, the transcript
# sentence model, and the YOLO-World detector), regardless of which optional
# features are enabled in .env, so nothing downloads mid-pipeline or on the
# first search. Cached files are skipped, so re-running setup is cheap.
echo "=== Downloading all model weights (one-time, ~1.3 GB) ==="
uv run python scripts/download_models.py

# OCR search (see README) is opt-in and off by default -- its backend is a
# system binary (not a pip package: pytesseract is just a thin wrapper
# around it), so `uv sync` alone can't install it. Only check/warn if the
# user has actually enabled it, same reasoning as the YOLO-World check above.
if [ -f .env ] && grep -qE '^ENABLE_OCR_SEARCH=true' .env; then
    if ! command -v tesseract >/dev/null 2>&1; then
        echo "warning: ENABLE_OCR_SEARCH=true but the 'tesseract' binary isn't installed." >&2
        echo "  Install it (e.g. 'sudo apt install tesseract-ocr' on Debian/Ubuntu) before indexing." >&2
    fi
fi

echo
echo "Setup complete. Next steps:"
echo "  1. Upload source videos to the raw-videos bucket on your S3/Floci endpoint"
echo "  2. Run ./run.sh"
