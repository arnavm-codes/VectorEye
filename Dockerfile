# Match the Python version in .python-version / pyproject.toml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Bake model weights into the image (script already exists in the repo).
RUN python scripts/download_models.py

EXPOSE 9100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9100"]
