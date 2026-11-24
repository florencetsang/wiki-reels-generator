"""Shared types and small helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests


Row = dict[str, Any]
Statement = tuple[str, tuple[Any, ...]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def cloudflare_result(response: requests.Response, action: str) -> Any:
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if not data.get("success"):
        raise RuntimeError(
            f"{action} failed (HTTP {response.status_code}): {data.get('errors')}"
        )
    return data["result"]
