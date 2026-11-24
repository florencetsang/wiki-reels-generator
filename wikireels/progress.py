"""Progress database: local SQLite or Cloudflare D1, plus the pipeline queries."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import requests

from .common import Row, Statement, cloudflare_result, utc_now_iso
from .config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, D1_DATABASE_ID
from .wikipedia import VitalTopic


# Statements per D1 HTTP request; each request must finish within 30 seconds.
D1_BATCH_SIZE = 200

SCHEMA = """
CREATE TABLE IF NOT EXISTS discoveries (
    level         INTEGER PRIMARY KEY,
    article_count INTEGER NOT NULL,
    completed_at  TEXT NOT NULL
);

-- status: pending -> summarized -> embedded -> done | skipped
CREATE TABLE IF NOT EXISTS articles (
    title           TEXT PRIMARY KEY,
    vital_level     INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    pageid          INTEGER,
    resolved_title  TEXT,
    canonical_url   TEXT,
    revision_id     INTEGER,
    summary         TEXT,
    categories      TEXT,
    embedding_model TEXT,
    embedding_dim   INTEGER,
    embedding       TEXT,
    r2_reel_bucket  TEXT,
    r2_reel_key     TEXT,
    r2_reel_uploaded_at TEXT,
    vectorized_at   TEXT,
    vector_index    TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_level_status
    ON articles (vital_level, status);

CREATE TABLE IF NOT EXISTS sections (
    title         TEXT NOT NULL REFERENCES articles (title),
    position      INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    content       TEXT NOT NULL,
    output_path   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    r2_bucket     TEXT,
    r2_key        TEXT,
    r2_uploaded_at TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (title, position)
);
"""


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------

class LocalSQLite:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")

    def script(self, sql: str) -> None:
        self.conn.executescript(sql)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
        with self.conn:
            return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def batch(self, statements: list[Statement]) -> None:
        with self.conn:
            for sql, params in statements:
                self.conn.execute(sql, params)


class D1Database:
    """Cloudflare D1 over its HTTP query API."""

    def __init__(self) -> None:
        required = {
            "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
            "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
            "D1_DATABASE_ID": D1_DATABASE_ID,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("Missing D1 configuration: " + ", ".join(missing))

        self.url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{CLOUDFLARE_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query"
        )
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {CLOUDFLARE_API_TOKEN}"

    def _post(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.session.post(self.url, json=body, timeout=60)
        return cloudflare_result(response, "D1 query")

    def script(self, sql: str) -> None:
        self._post({"sql": sql})

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
        result = self._post({"sql": sql, "params": list(params)})
        return result[0].get("results") or []

    def batch(self, statements: list[Statement]) -> None:
        # D1 runs each batch request as a single transaction.
        for start in range(0, len(statements), D1_BATCH_SIZE):
            self._post(
                {
                    "batch": [
                        {"sql": sql, "params": list(params)}
                        for sql, params in statements[start : start + D1_BATCH_SIZE]
                    ]
                }
            )


def open_progress_backend(path: Path, local: bool) -> LocalSQLite | D1Database:
    if not local and (D1_DATABASE_ID or CLOUDFLARE_API_TOKEN):
        logging.info("Progress database: Cloudflare D1 %s", D1_DATABASE_ID)
        return D1Database()
    logging.info("Progress database: local SQLite %s", path)
    return LocalSQLite(path)


# -----------------------------------------------------------------------------
# Pipeline queries
# -----------------------------------------------------------------------------

class ProgressDB:
    def __init__(self, backend: LocalSQLite | D1Database) -> None:
        self.db = backend
        self.db.script(SCHEMA)
        self._ensure_upload_tracking_columns()

    def _ensure_upload_tracking_columns(self) -> None:
        columns_by_table = {
            "articles": {
                "r2_reel_bucket": "TEXT",
                "r2_reel_key": "TEXT",
                "r2_reel_uploaded_at": "TEXT",
                "vectorized_at": "TEXT",
                "vector_index": "TEXT",
                "vital_field": "TEXT",
                "vital_subfield": "TEXT",
            },
            "sections": {
                "r2_bucket": "TEXT",
                "r2_key": "TEXT",
                "r2_uploaded_at": "TEXT",
            },
        }
        for table, columns in columns_by_table.items():
            existing = {
                row["name"] for row in self.db.query(f"PRAGMA table_info({table})")
            }
            for name, column_type in columns.items():
                if name not in existing:
                    self.db.query(
                        f"ALTER TABLE {table} ADD COLUMN {name} {column_type}"
                    )

    def _one(self, sql: str, params: tuple[Any, ...]) -> Row | None:
        rows = self.db.query(sql, params)
        return rows[0] if rows else None

    # --- discovery ---

    def is_discovered(self, level: int) -> bool:
        return self._one(
            "SELECT 1 FROM discoveries WHERE level = ?", (level,)
        ) is not None

    def save_discovery(self, level: int, titles: dict[str, VitalTopic]) -> None:
        now = utc_now_iso()
        self.db.batch(
            [
                (
                    "INSERT INTO articles (title, vital_level, vital_field, "
                    "vital_subfield, updated_at) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (title) DO UPDATE SET "
                    "vital_field = COALESCE(excluded.vital_field, vital_field), "
                    "vital_subfield = COALESCE(excluded.vital_subfield, vital_subfield)",
                    (title, level, topic.field, topic.subfield, now),
                )
                for title, topic in titles.items()
            ]
        )
        self.db.query(
            "INSERT OR REPLACE INTO discoveries VALUES (?, ?, ?)",
            (level, len(titles), now),
        )

    # --- articles ---

    def get_article(self, title: str) -> Row:
        return self._one("SELECT * FROM articles WHERE title = ?", (title,))

    def pending_articles(
        self,
        level: int,
        include_images: bool,
        max_attempts: int,
        limit: int,
    ) -> list[Row]:
        statuses = ["pending", "summarized"]
        if include_images:
            statuses.append("embedded")
        placeholders = ",".join("?" * len(statuses))
        return self.db.query(
            f"SELECT * FROM articles WHERE vital_level <= ? "
            f"AND status IN ({placeholders}) AND attempts < ? "
            f"ORDER BY title COLLATE NOCASE LIMIT {int(limit) if limit > 0 else -1}",
            (level, *statuses, max_attempts),
        )

    def update_article(self, title: str, **fields: Any) -> None:
        fields["updated_at"] = utc_now_iso()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self.db.query(
            f"UPDATE articles SET {assignments} WHERE title = ?",
            (*fields.values(), title),
        )

    def record_failure(self, title: str, error: str) -> None:
        self.db.query(
            "UPDATE articles SET attempts = attempts + 1, error = ?, "
            "updated_at = ? WHERE title = ?",
            (error, utc_now_iso(), title),
        )

    # --- sections ---

    def get_sections(self, title: str) -> list[Row]:
        return self.db.query(
            "SELECT * FROM sections WHERE title = ? ORDER BY position",
            (title,),
        )

    def insert_sections(
        self,
        title: str,
        sections: list[tuple[int, str, str, str]],
    ) -> None:
        now = utc_now_iso()
        self.db.batch(
            [
                (
                    "INSERT OR IGNORE INTO sections (title, position, section_title, "
                    "content, output_path, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (title, *section, now),
                )
                for section in sections
            ]
        )

    def mark_section_done(self, title: str, position: int) -> None:
        self.db.query(
            "UPDATE sections SET status = 'done', updated_at = ? "
            "WHERE title = ? AND position = ?",
            (utc_now_iso(), title, position),
        )

    def mark_section_uploaded(
        self, title: str, position: int, bucket: str, key: str
    ) -> None:
        now = utc_now_iso()
        self.db.query(
            "UPDATE sections SET r2_bucket = ?, r2_key = ?, r2_uploaded_at = ?, "
            "updated_at = ? WHERE title = ? AND position = ?",
            (bucket, key, now, now, title, position),
        )

    # --- vectors ---

    def unvectorized_articles(self, index_name: str, limit: int) -> list[Row]:
        return self.db.query(
            "SELECT title, pageid, resolved_title, vital_level, vital_field, "
            "canonical_url, r2_reel_key, embedding FROM articles "
            "WHERE status = 'done' AND vector_index IS NOT ? "
            f"ORDER BY title LIMIT {int(limit)}",
            (index_name,),
        )

    def mark_vectorized(self, titles: list[str], index_name: str) -> None:
        now = utc_now_iso()
        self.db.batch(
            [
                (
                    "UPDATE articles SET vectorized_at = ?, vector_index = ?, "
                    "updated_at = ? WHERE title = ?",
                    (now, index_name, now, title),
                )
                for title in titles
            ]
        )

    # --- reporting ---

    def print_status(self, level: int) -> None:
        discoveries = self.db.query(
            "SELECT * FROM discoveries WHERE level <= ? ORDER BY level", (level,)
        )
        if len(discoveries) < level:
            print(f"Level {level}: not fully discovered yet.")
            return

        print(
            f"Level {level} (levels 1-{level}): "
            f"{sum(d['article_count'] for d in discoveries)} articles "
            f"(discovered {discoveries[-1]['completed_at']})"
        )
        print("Articles by status:")
        for row in self.db.query(
            "SELECT status, COUNT(*) AS n, SUM(error IS NOT NULL) AS with_error "
            "FROM articles WHERE vital_level <= ? GROUP BY status ORDER BY status",
            (level,),
        ):
            print(f"  {row['status']:<11} {row['n']:>6}  (with error: {row['with_error']})")

        print("Infographic sections by status:")
        for row in self.db.query(
            "SELECT s.status, COUNT(*) AS n FROM sections s "
            "JOIN articles a ON a.title = s.title "
            "WHERE a.vital_level <= ? GROUP BY s.status ORDER BY s.status",
            (level,),
        ):
            print(f"  {row['status']:<11} {row['n']:>6}")
