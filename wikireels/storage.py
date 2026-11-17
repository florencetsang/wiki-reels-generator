"""Infographic files: local paths, WebP encoding and Cloudflare R2 upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from PIL import Image
from wikiinfo import safe_filename

from .common import Row, to_json_bytes
from .config import (
    R2_ACCESS_KEY_ID,
    R2_BUCKET,
    R2_ENDPOINT_URL,
    R2_SECRET_ACCESS_KEY,
)


class R2Store:
    def __init__(self) -> None:
        required = {
            "R2_BUCKET": R2_BUCKET,
            "R2_ENDPOINT_URL": R2_ENDPOINT_URL,
            "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
            "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing R2 configuration: " + ", ".join(missing)
            )

        self.bucket = R2_BUCKET
        self.client = boto3.client(
            service_name="s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            raise RuntimeError(
                f"R2 bucket {self.bucket!r} is not accessible at "
                f"{R2_ENDPOINT_URL} ({exc}); create it or fix R2_BUCKET"
            ) from exc

    def object_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def put_json(self, key: str, value: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=to_json_bytes(value),
            ContentType="application/json; charset=utf-8",
        )

    def put_image(self, key: str, content: bytes, mime_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=mime_type,
        )


def infographic_key(pageid: int, position: int, output_path: Path) -> str:
    return f"reels/{pageid}/infographics/{position:03d}{output_path.suffix}"


def topic_dir(output_dir: Path, row: Row) -> Path:
    # pageid prefix: distinct titles can share a safe_filename ("Cell (biology)", "Cell biology").
    title = row["resolved_title"] or row["title"]
    return output_dir / f"{row['pageid']}-{safe_filename(title)}"


def ensure_webp(path: Path) -> None:
    # google-genai's Image.save() writes the raw (usually PNG) bytes regardless of extension.
    with Image.open(path) as image:
        if image.format == "WEBP":
            return
        partial = path.with_suffix(".partial")
        image.save(partial, "WEBP", quality=90)
    partial.replace(path)
