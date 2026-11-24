"""Vector indexes (Cloudflare Vectorize or local SQLite) and syncing."""

from __future__ import annotations

import json
import logging
import sqlite3
from array import array
from pathlib import Path
from typing import Any

import requests

from .common import Row, cloudflare_result
from .config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, VECTORIZE_INDEX
from .embeddings import decode_embedding
from .progress import ProgressDB


# Vectors per upsert request; Cloudflare recommends at most 5,000.
VECTORIZE_BATCH_SIZE = 500


class VectorizeIndex:
    """Upserts vectors through the Vectorize v2 HTTP API."""

    def __init__(self) -> None:
        required = {
            "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
            "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
            "VECTORIZE_INDEX": VECTORIZE_INDEX,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("Missing Vectorize configuration: " + ", ".join(missing))

        self.name = VECTORIZE_INDEX
        self.url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{CLOUDFLARE_ACCOUNT_ID}/vectorize/v2/indexes/{VECTORIZE_INDEX}/upsert"
        )
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {CLOUDFLARE_API_TOKEN}"

    def upsert(self, vectors: list[dict[str, Any]]) -> str:
        ndjson = "\n".join(json.dumps(vector, ensure_ascii=False) for vector in vectors)
        response = self.session.post(
            self.url,
            files={"vectors": ("vectors.ndjson", ndjson.encode("utf-8"), "application/x-ndjson")},
            timeout=120,
        )
        return cloudflare_result(response, "Vectorize upsert").get("mutationId", "")


class LocalVectorIndex:
    """Local stand-in for Cloudflare Vectorize and the Worker's /related endpoint."""

    def __init__(self, path: Path) -> None:
        self.name = f"local:{path.name}"
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "id TEXT PRIMARY KEY, title TEXT NOT NULL, vital_level INTEGER, "
            "metadata TEXT NOT NULL, vector BLOB NOT NULL)"
        )

    def upsert(self, vectors: list[dict[str, Any]]) -> str:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        vector["id"],
                        vector["metadata"]["title"],
                        vector["metadata"].get("vital_level"),
                        json.dumps(vector["metadata"], ensure_ascii=False),
                        array("f", vector["values"]).tobytes(),
                    )
                    for vector in vectors
                ],
            )
        return ""

    def related(self, pageid_or_title: str, top_k: int, max_level: int) -> list[Row]:
        import numpy as np

        target = self.conn.execute(
            "SELECT * FROM vectors WHERE id = ? OR title = ? COLLATE NOCASE",
            (pageid_or_title, pageid_or_title),
        ).fetchone()
        if target is None:
            raise LookupError(f"{pageid_or_title!r} is not in the local vector index")

        rows = self.conn.execute(
            "SELECT id, metadata, vector FROM vectors WHERE vital_level <= ? AND id != ?",
            (max_level, target["id"]),
        ).fetchall()
        if not rows:
            return []

        # Embeddings are normalized, so the dot product is the cosine similarity.
        matrix = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=np.float32)
        scores = matrix.reshape(len(rows), -1) @ np.frombuffer(target["vector"], dtype=np.float32)
        return [
            {
                "pageid": int(rows[i]["id"]),
                "score": round(float(scores[i]), 4),
                **json.loads(rows[i]["metadata"]),
            }
            for i in np.argsort(-scores)[:top_k]
        ]


def sync_vectors(db: ProgressDB, index: VectorizeIndex | LocalVectorIndex) -> int:
    """Upsert embeddings of finished reels that are not in this index yet."""
    total = 0
    while rows := db.unvectorized_articles(index.name, VECTORIZE_BATCH_SIZE):
        vectors = []
        for row in rows:
            metadata = {
                "title": row["resolved_title"] or row["title"],
                "vital_level": row["vital_level"],
                "vital_field": row["vital_field"],
                "url": row["canonical_url"],
                "reel_key": row["r2_reel_key"],
            }
            vectors.append(
                {
                    "id": str(row["pageid"]),
                    "values": decode_embedding(row["embedding"]),
                    "metadata": {k: v for k, v in metadata.items() if v is not None},
                }
            )
        mutation_id = index.upsert(vectors)
        db.mark_vectorized([row["title"] for row in rows], index.name)
        total += len(rows)
        logging.info(
            "Upserted %d vectors to %s%s",
            len(rows),
            index.name,
            f" (mutation {mutation_id})" if mutation_id else "",
        )
    return total
