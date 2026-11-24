"""Settings read from the environment / .env file."""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "WikiToInfographicBot/1.0 (contact@example.com)",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def cloudflare_env(name: str) -> str:
    # Values copied unchanged from .env.example count as unset.
    value = os.getenv(name, "").strip()
    return "" if value == "replace_me" else value


LOCAL_ONLY = os.getenv("LOCAL_ONLY", "").strip().lower() in {"1", "true", "yes"}

R2_BUCKET = cloudflare_env("R2_BUCKET")
R2_ACCOUNT_ID = cloudflare_env("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = cloudflare_env("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = cloudflare_env("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = cloudflare_env("R2_ENDPOINT_URL") or (
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    if R2_ACCOUNT_ID
    else ""
)

CLOUDFLARE_ACCOUNT_ID = cloudflare_env("CLOUDFLARE_ACCOUNT_ID") or R2_ACCOUNT_ID
CLOUDFLARE_API_TOKEN = cloudflare_env("CLOUDFLARE_API_TOKEN")
D1_DATABASE_ID = cloudflare_env("D1_DATABASE_ID")
VECTORIZE_INDEX = cloudflare_env("VECTORIZE_INDEX")
