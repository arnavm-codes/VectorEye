"""Thin S3 wrapper (Floci locally). S3 is the system of record for raw videos and
clips; all heavy processing (ffmpeg, cv2, whisper) runs on temp copies."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import (
    CLIPS_BUCKET,
    PRESIGN_EXPIRY_SECONDS,
    RAW_VIDEOS_BUCKET,
    S3_ACCESS_KEY,
    S3_ENDPOINT,
    S3_PUBLIC_ENDPOINT,
    S3_REGION,
    S3_SECRET_KEY,
    S3_VERIFY_SSL,
    VIDEO_EXTENSIONS,
)

_CONFIG = Config(signature_version="s3v4", s3={"addressing_style": "path"})
_client = None
_public_client = None


# Health checks must answer quickly even when the S3 host is down or firewalled
# (boto3's default 60s connect timeout + retries would make /health hang).
_HEALTH_CONFIG = Config(
    signature_version="s3v4",
    s3={"addressing_style": "path"},
    connect_timeout=3,
    read_timeout=5,
    retries={"max_attempts": 1},
)


def _make_client(endpoint: str, config: Config = _CONFIG):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        verify=S3_VERIFY_SSL,
        config=config,
    )


def get_s3():
    global _client
    if _client is None:
        _client = _make_client(S3_ENDPOINT)
    return _client


def _get_public_s3():
    """Client used only to sign URLs against the externally-reachable endpoint."""
    global _public_client
    if _public_client is None:
        _public_client = _make_client(S3_PUBLIC_ENDPOINT)
    return _public_client


def list_keys(bucket: str, suffixes: set[str] | None = None, prefix: str = "") -> list[str]:
    keys = []
    for page in get_s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if suffixes and Path(key).suffix.lower() not in suffixes:
                continue
            keys.append(key)
    return sorted(keys)


def list_raw_videos(prefix: str = "") -> list[str]:
    return list_keys(RAW_VIDEOS_BUCKET, VIDEO_EXTENSIONS, prefix)


def raw_video_exists(key: str) -> bool:
    try:
        get_s3().head_object(Bucket=RAW_VIDEOS_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def raw_video_fingerprint(key: str) -> list:
    """(ETag, size) of a raw video object -- what change detection compares to
    decide whether a video needs re-processing. Raises ClientError if missing."""
    head = get_s3().head_object(Bucket=RAW_VIDEOS_BUCKET, Key=key)
    return [head["ETag"].strip('"'), head["ContentLength"]]


def list_clips() -> list[str]:
    return list_keys(CLIPS_BUCKET, {".mp4"})


def download(bucket: str, key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    get_s3().download_file(bucket, key, str(dest))
    return dest


def upload_clip(local_path: Path, key: str | None = None) -> str:
    key = key or local_path.name
    get_s3().upload_file(str(local_path), CLIPS_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return key


def clip_exists(key: str) -> bool:
    try:
        get_s3().head_object(Bucket=CLIPS_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def delete_clip(key: str) -> None:
    get_s3().delete_object(Bucket=CLIPS_BUCKET, Key=key)


def presigned_clip_url(key: str) -> str:
    return _get_public_s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": CLIPS_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRY_SECONDS,
    )


def open_clip_stream(key: str):
    """Returns boto3's get_object response (Body is a streaming file-like)."""
    return get_s3().get_object(Bucket=CLIPS_BUCKET, Key=key)


@contextmanager
def local_clip(key: str):
    """Download a clip to a temp file and yield its Path; deleted on exit."""
    with tempfile.TemporaryDirectory(prefix="vectoreye_") as tmp:
        yield download(CLIPS_BUCKET, key, Path(tmp) / Path(key).name)


def ensure_buckets() -> list[str]:
    """Create the raw-videos and clips buckets if they don't exist yet.
    Idempotent: existing buckets are left untouched. Returns the names created."""
    s3 = get_s3()
    created = []
    for bucket in (RAW_VIDEOS_BUCKET, CLIPS_BUCKET):
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            s3.create_bucket(Bucket=bucket)
            created.append(bucket)
    return created


def s3_healthy() -> tuple[bool, str]:
    try:
        buckets = {b["Name"] for b in _make_client(S3_ENDPOINT, _HEALTH_CONFIG).list_buckets().get("Buckets", [])}
        missing = {RAW_VIDEOS_BUCKET, CLIPS_BUCKET} - buckets
        if missing:
            return False, f"missing buckets: {sorted(missing)}"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - health check must never raise
        return False, str(exc)
