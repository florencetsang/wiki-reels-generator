"""Command-line arguments and the run loop."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from google.genai import errors
from wikiinfo import create_client

from .config import (
    EMBEDDING_MODEL_NAME,
    LOCAL_ONLY,
    R2_ACCESS_KEY_ID,
    R2_BUCKET,
    R2_SECRET_ACCESS_KEY,
    VECTORIZE_INDEX,
)
from .embeddings import Embedder
from .pipeline import backfill_labels, process_article
from .progress import D1Database, ProgressDB, open_progress_backend
from .storage import R2Store
from .vectors import LocalVectorIndex, VectorizeIndex, sync_vectors
from .wikipedia import WikipediaClient


# Quota exhaustion / access denied affect every article, so stop the run
# instead of burning per-article retry attempts.
FATAL_GEMINI_CODES = {403, 429}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable infographic generation for Wikipedia Vital Articles."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=4,
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
        default=LOCAL_ONLY,
        help="Ignore Cloudflare config: local SQLite, no R2, local vector index "
        "(default when LOCAL_ONLY=true)",
    )
    parser.add_argument(
        "--vectors-db",
        type=Path,
        default=Path("vectors.sqlite3"),
        help="Local vector index, used when Cloudflare Vectorize is not configured "
        "(default: vectors.sqlite3)",
    )
    parser.add_argument(
        "--related",
        metavar="PAGEID_OR_TITLE",
        help="Print reels similar to this one (levels <= --level) from the local "
        "vector index and exit",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results for --related (default: 10)",
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="Re-scan Wikipedia categories for newly added articles and "
        "to refresh vital_field / vital_subfield labels",
    )
    parser.add_argument(
        "--backfill-labels",
        action="store_true",
        help="Add vital_field / vital_subfield to finished reels (reel JSON and "
        "vector metadata) without calling Gemini or re-embedding, then exit",
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
    if args.log_level != "DEBUG":
        for name in ("httpx", "httpcore", "huggingface_hub", "wikipediaapi"):
            logging.getLogger(name).setLevel(logging.WARNING)

    try:
        db = ProgressDB(open_progress_backend(args.db, args.local))
        r2 = (
            R2Store()
            if not args.local
            and not args.no_images
            and (R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY)
            else None
        )
        vectorize = (
            VectorizeIndex()
            if not args.local and VECTORIZE_INDEX
            else LocalVectorIndex(args.vectors_db)
        )
    except RuntimeError as exc:
        logging.error("%s. Use --local to run without Cloudflare.", exc)
        return 2

    if args.status:
        db.print_status(args.level)
        return 0

    if args.related:
        try:
            matches = LocalVectorIndex(args.vectors_db).related(
                args.related, args.top_k, args.level
            )
        except LookupError as exc:
            logging.error("%s. Finished reels are added at the end of each run.", exc)
            return 1
        print(json.dumps(matches, ensure_ascii=False, indent=2))
        return 0

    if args.backfill_labels:
        if r2 is None and not args.local:
            logging.error("--backfill-labels needs R2 credentials to rewrite reel JSON.")
            return 2
        count = backfill_labels(db, WikipediaClient(), r2, args.level, args.output_dir)
        logging.info("Reel JSON rewritten with labels: %d", count)
        logging.info("Vectors re-upserted with labels: %d", sync_vectors(db, vectorize))
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
    logging.info("Vector sync: %s", vectorize.name)

    wiki = WikipediaClient()
    # Wikipedia categories are per exact level; the level-N list includes levels 1..N.
    for level in range(1, args.level + 1):
        if args.rediscover or not db.is_discovered(level):
            titles = wiki.discover_vital_titles(level)
            db.save_discovery(level, titles)
            logging.info("Discovered %d level-%d vital articles", len(titles), level)

    gemini = None if args.no_images else create_client()
    embedder = Embedder(EMBEDDING_MODEL_NAME)

    rows = db.pending_articles(
        args.level,
        include_images=not args.no_images,
        max_attempts=args.max_attempts,
        limit=args.limit,
    )
    logging.info("Articles to process this run: %d", len(rows))

    exit_code = 0
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
                exit_code = 1
                break
            logging.exception("Failed: %s", title)
            db.record_failure(title, f"ClientError: {exc}")
        except Exception as exc:
            logging.exception("Failed: %s", title)
            db.record_failure(title, f"{type(exc).__name__}: {exc}")

    try:
        logging.info("Vectors synced this run: %d", sync_vectors(db, vectorize))
    except Exception:
        logging.exception("Vector sync failed; it will be retried next run.")
        exit_code = 1

    db.print_status(args.level)
    return exit_code
