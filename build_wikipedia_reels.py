#!/usr/bin/env python3
"""
Generate infographics for every Wikipedia Vital Article at a given level.

Per-article pipeline, checkpointed in SQLite (Cloudflare D1 when configured,
otherwise a local file) so a run can be stopped (Ctrl+C, crash, quota
exhaustion) and resumed at any time:
  1. discover  - list level-X vital articles from Wikipedia categories
  2. summarize - fetch lead summary, categories, page ID and revision
  3. embed     - compute a local sentence-transformers embedding
  4. images    - one infographic per section, using wikiinfo.py logic
  5. upload    - infographics and reel JSON to Cloudflare R2 when configured:
                   reels/<pageid>.json
                   reels/<pageid>/infographics/001.webp ...
                 otherwise reel JSON is written to <output-dir>/<topic>/reel.json

Usage:
  python build_wikipedia_reels.py                  # level 3, resumes automatically
  python build_wikipedia_reels.py --level 4 --limit 20
  python build_wikipedia_reels.py --no-images      # summaries + embeddings only
  python build_wikipedia_reels.py --local          # ignore Cloudflare config
  python build_wikipedia_reels.py --status         # print progress and exit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from google.genai import errors

from wikiinfo import (
    build_infographic_prompt,
    create_client,
    fetch_wikipedia_sections,
    generate_infographic_image,
    safe_filename,
)


load_dotenv()

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "WikiToInfographicBot/1.0 (contact@example.com)",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
WIKIPEDIA_DELAY_SECONDS = 0.10


def cloudflare_env(name: str) -> str:
    # Values copied unchanged from .env.example count as unset.
    value = os.getenv(name, "").strip()
    return "" if value == "replace_me" else value


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
# Statements per D1 HTTP request; each request must finish within 30 seconds.
D1_BATCH_SIZE = 200

# Quota exhaustion / access denied affect every article, so stop the run
# instead of burning per-article retry attempts.
FATAL_GEMINI_CODES = {403, 429}

ARTICLE_NS = 0
TALK_NS = 1
CATEGORY_NS = 14

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
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (title, position)
);
"""


Row = dict[str, Any]
Statement = tuple[str, tuple[Any, ...]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


# -----------------------------------------------------------------------------
# Progress database backends
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
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if not data.get("success"):
            raise RuntimeError(
                f"D1 query failed (HTTP {response.status_code}): {data.get('errors')}"
            )
        return data["result"]

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
# Progress database
# -----------------------------------------------------------------------------

class ProgressDB:
    def __init__(self, backend: LocalSQLite | D1Database) -> None:
        self.db = backend
        self.db.script(SCHEMA)

    def _one(self, sql: str, params: tuple[Any, ...]) -> Row | None:
        rows = self.db.query(sql, params)
        return rows[0] if rows else None

    def is_discovered(self, level: int) -> bool:
        return self._one(
            "SELECT 1 FROM discoveries WHERE level = ?", (level,)
        ) is not None

    def save_discovery(self, level: int, titles: list[str]) -> None:
        now = utc_now_iso()
        self.db.batch(
            [
                (
                    "INSERT OR IGNORE INTO articles (title, vital_level, updated_at) "
                    "VALUES (?, ?, ?)",
                    (title, level, now),
                )
                for title in titles
            ]
        )
        self.db.query(
            "INSERT OR REPLACE INTO discoveries VALUES (?, ?, ?)",
            (level, len(titles), now),
        )

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
            f"SELECT * FROM articles WHERE vital_level = ? "
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

    def print_status(self, level: int) -> None:
        discovery = self._one(
            "SELECT * FROM discoveries WHERE level = ?", (level,)
        )
        if discovery is None:
            print(f"Level {level}: not discovered yet.")
            return

        print(
            f"Level {level}: {discovery['article_count']} articles "
            f"(discovered {discovery['completed_at']})"
        )
        print("Articles by status:")
        for row in self.db.query(
            "SELECT status, COUNT(*) AS n, SUM(error IS NOT NULL) AS with_error "
            "FROM articles WHERE vital_level = ? GROUP BY status ORDER BY status",
            (level,),
        ):
            print(f"  {row['status']:<11} {row['n']:>6}  (with error: {row['with_error']})")

        print("Infographic sections by status:")
        for row in self.db.query(
            "SELECT s.status, COUNT(*) AS n FROM sections s "
            "JOIN articles a ON a.title = s.title "
            "WHERE a.vital_level = ? GROUP BY s.status ORDER BY s.status",
            (level,),
        ):
            print(f"  {row['status']:<11} {row['n']:>6}")


# -----------------------------------------------------------------------------
# Wikipedia
# -----------------------------------------------------------------------------

class WikipediaClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def api(self, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(
            WIKIPEDIA_API,
            params={**params, "format": "json", "formatversion": "2"},
            timeout=45,
        )
        response.raise_for_status()
        time.sleep(WIKIPEDIA_DELAY_SECONDS)
        return response.json()

    def discover_vital_titles(self, level: int) -> list[str]:
        """
        Vital-article categories are populated by talk-page banners, so members
        are mostly "Talk:<Article>" pages. Subcategories are followed only when
        they belong to the same level.
        """
        level_marker = f"level-{level} vital"
        queue = [f"Category:Wikipedia level-{level} vital articles"]
        seen: set[str] = set()
        titles: set[str] = set()

        while queue:
            category = queue.pop()
            if category in seen:
                continue
            seen.add(category)
            logging.info("Scanning %s (%d titles so far)", category, len(titles))

            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": "max",
                "cmprop": "title",
                "cmnamespace": f"{ARTICLE_NS}|{TALK_NS}|{CATEGORY_NS}",
            }
            continuation: dict[str, Any] = {}

            while True:
                result = self.api({**params, **continuation})
                for member in result.get("query", {}).get("categorymembers", []):
                    ns = member.get("ns")
                    title = member["title"]
                    if ns == CATEGORY_NS:
                        if level_marker in title.casefold():
                            queue.append(title)
                    elif ns == TALK_NS:
                        titles.add(title.removeprefix("Talk:"))
                    elif ns == ARTICLE_NS:
                        titles.add(title)

                if "continue" not in result:
                    break
                continuation = result["continue"]

        return sorted(titles, key=str.casefold)

    def get_article_metadata(self, title: str) -> dict[str, Any] | None:
        result = self.api(
            {
                "action": "query",
                "redirects": "1",
                "prop": "extracts|categories|info|revisions",
                "titles": title,
                "exintro": "1",
                "explaintext": "1",
                "cllimit": "max",
                "clshow": "!hidden",
                "inprop": "url",
                "rvprop": "ids",
            }
        )

        pages = result.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or pages[0].get("invalid"):
            return None

        page = pages[0]
        categories = sorted(
            {
                category["title"].removeprefix("Category:")
                for category in page.get("categories", [])
            }
        )
        revisions = page.get("revisions") or [{}]

        return {
            "pageid": int(page["pageid"]),
            "resolved_title": page["title"],
            "canonical_url": page.get("canonicalurl"),
            "revision_id": revisions[0].get("revid"),
            "summary": (page.get("extract") or "").strip(),
            "categories": json.dumps(categories, ensure_ascii=False),
        }


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, summary: str, categories: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logging.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

        source = (
            f"Summary:\n{summary}\n\nWikipedia categories:\n"
            + "\n".join(f"- {category}" for category in categories)
        )
        vector = self._model.encode(
            source,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vector]


# -----------------------------------------------------------------------------
# Cloudflare R2
# -----------------------------------------------------------------------------

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


def decode_embedding(value: str | bytes) -> list[float]:
    if isinstance(value, bytes):
        # Local databases created before embeddings were stored as JSON.
        vector = array("f")
        vector.frombytes(value)
        return list(vector)
    return json.loads(value)


def make_reel_object(row: Row, sections: list[Row]) -> dict[str, Any]:
    return {
        "pageid": row["pageid"],
        "title": row["resolved_title"] or row["title"],
        "canonical_url": row["canonical_url"],
        "revision_id": row["revision_id"],
        "vital_level": row["vital_level"],
        "summary": row["summary"],
        "categories": json.loads(row["categories"]),
        "embedding": {
            "model": row["embedding_model"],
            "dim": row["embedding_dim"],
            "vector": decode_embedding(row["embedding"]),
        },
        "infographics": [
            {
                "page": section["position"],
                "section": section["section_title"],
                "r2_key": infographic_key(
                    row["pageid"], section["position"], Path(section["output_path"])
                ),
                "mime_type": "image/webp",
            }
            for section in sections
        ],
        "updated_at": utc_now_iso(),
    }


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def generate_article_infographics(
    db: ProgressDB,
    gemini,
    r2: R2Store | None,
    row: Row,
    output_dir: Path,
    max_sections: int,
) -> None:
    title = row["title"]
    article_title = row["resolved_title"] or title

    sections = db.get_sections(title)
    if not sections:
        fetched = fetch_wikipedia_sections(article_title)
        if max_sections > 0:
            fetched = fetched[:max_sections]
        topic_dir = output_dir / safe_filename(article_title)
        db.insert_sections(
            title,
            [
                (
                    position,
                    section_title,
                    content,
                    str(topic_dir / f"{position:02d}-{safe_filename(section_title)}.webp"),
                )
                for position, (section_title, content) in enumerate(fetched, start=1)
            ],
        )
        sections = db.get_sections(title)

    for section in sections:
        if section["status"] == "done":
            continue

        output_path = Path(section["output_path"])
        # The image may have been saved right before an interruption.
        if not output_path.exists():
            logging.info(
                "  [%d/%d] Generating: %s",
                section["position"],
                len(sections),
                section["section_title"],
            )
            prompt = build_infographic_prompt(
                article_title,
                section["section_title"],
                section["content"],
            )
            generate_infographic_image(gemini, prompt, output_path)

        if r2:
            r2.put_image(
                infographic_key(row["pageid"], section["position"], output_path),
                output_path.read_bytes(),
                "image/webp",
            )

        db.mark_section_done(title, section["position"])


def process_article(
    db: ProgressDB,
    wiki: WikipediaClient,
    embedder: Embedder,
    gemini,
    r2: R2Store | None,
    row: Row,
    args: argparse.Namespace,
) -> str:
    title = row["title"]

    if row["status"] == "pending":
        metadata = wiki.get_article_metadata(title)
        if metadata is None:
            db.update_article(title, status="skipped", error="article missing")
            return "skipped"
        if not metadata["summary"]:
            db.update_article(title, status="skipped", error="no summary", **metadata)
            return "skipped"
        db.update_article(title, status="summarized", **metadata)
        row = db.get_article(title)

    if row["status"] == "summarized":
        vector = embedder.embed(row["summary"], json.loads(row["categories"]))
        db.update_article(
            title,
            status="embedded",
            embedding_model=embedder.model_name,
            embedding_dim=len(vector),
            # JSON text because the D1 HTTP API cannot bind BLOB parameters.
            embedding=json.dumps(vector),
        )
        row = db.get_article(title)

    if args.no_images:
        return row["status"]

    generate_article_infographics(
        db, gemini, r2, row, args.output_dir, args.max_sections
    )
    reel = make_reel_object(row, db.get_sections(title))
    if r2:
        r2.put_json(f"reels/{row['pageid']}.json", reel)
    else:
        reel_path = (
            args.output_dir / safe_filename(row["resolved_title"] or title) / "reel.json"
        )
        reel_path.parent.mkdir(parents=True, exist_ok=True)
        reel_path.write_bytes(to_json_bytes(reel))
    db.update_article(title, status="done", error=None)
    return "done"


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable infographic generation for Wikipedia Vital Articles."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Vital article level (default: 3)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("vital_progress.sqlite3"),
        help="Local SQLite progress database, used when D1 is not configured "
        "(default: vital_progress.sqlite3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infographics"),
        help="Parent directory for topic folders (default: infographics)",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=10,
        help="Infographics per article, 0 = all sections (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum articles to process in this run, 0 = all",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Skip articles that have failed this many times (default: 3)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Only fetch summaries and compute embeddings",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Ignore Cloudflare config: use local SQLite and skip R2 upload",
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="Re-scan Wikipedia categories for newly added articles",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print progress for --level and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        db = ProgressDB(open_progress_backend(args.db, args.local))
        r2 = (
            R2Store()
            if not args.local
            and not args.no_images
            and (R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY)
            else None
        )
    except RuntimeError as exc:
        logging.error("%s. Use --local to run without Cloudflare.", exc)
        return 2

    if args.status:
        db.print_status(args.level)
        return 0

    if r2:
        logging.info("Infographic upload: R2 bucket %s", R2_BUCKET)
    elif not args.no_images:
        logging.info("Infographic upload: disabled, output stays in %s", args.output_dir)
        if isinstance(db.db, D1Database):
            logging.warning(
                "D1 is configured but R2 is not; articles will be marked done "
                "in D1 without uploaded infographics."
            )

    wiki = WikipediaClient()
    if args.rediscover or not db.is_discovered(args.level):
        titles = wiki.discover_vital_titles(args.level)
        db.save_discovery(args.level, titles)
        logging.info("Discovered %d level-%d vital articles", len(titles), args.level)

    gemini = None if args.no_images else create_client()
    embedder = Embedder(EMBEDDING_MODEL_NAME)

    rows = db.pending_articles(
        args.level,
        include_images=not args.no_images,
        max_attempts=args.max_attempts,
        limit=args.limit,
    )
    logging.info("Articles to process this run: %d", len(rows))

    for index, row in enumerate(rows, start=1):
        title = row["title"]
        logging.info("[%d/%d] %s (%s)", index, len(rows), title, row["status"])
        try:
            result = process_article(db, wiki, embedder, gemini, r2, row, args)
            logging.info("  -> %s", result)
        except errors.ClientError as exc:
            if exc.code in FATAL_GEMINI_CODES:
                db.update_article(title, error=f"ClientError: {exc}")
                logging.error(
                    "Gemini returned HTTP %s; stopping. Re-run later to resume.",
                    exc.code,
                )
                db.print_status(args.level)
                return 1
            logging.exception("Failed: %s", title)
            db.record_failure(title, f"ClientError: {exc}")
        except Exception as exc:
            logging.exception("Failed: %s", title)
            db.record_failure(title, f"{type(exc).__name__}: {exc}")

    db.print_status(args.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
