"""Per-article pipeline: summarize -> embed -> infographics -> reel JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from wikiinfo import (
    build_infographic_prompt,
    fetch_wikipedia_sections,
    generate_infographic_image,
    safe_filename,
)

from .common import Row, to_json_bytes, utc_now_iso
from .embeddings import Embedder, decode_embedding
from .progress import ProgressDB
from .storage import R2Store, ensure_webp, infographic_key, topic_dir
from .wikipedia import WikipediaClient


def make_reel_object(row: Row, sections: list[Row]) -> dict[str, Any]:
    return {
        "pageid": row["pageid"],
        "title": row["resolved_title"] or row["title"],
        "canonical_url": row["canonical_url"],
        "revision_id": row["revision_id"],
        "vital_level": row["vital_level"],
        "vital_field": row["vital_field"],
        "vital_subfield": row["vital_subfield"],
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
                "r2_bucket": section["r2_bucket"],
                "r2_key": section["r2_key"],
                "mime_type": "image/webp",
                **(
                    {"local_path": section["output_path"]}
                    if not section["r2_key"]
                    else {}
                ),
            }
            for section in sections
        ],
        "updated_at": utc_now_iso(),
    }


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
        article_dir = topic_dir(output_dir, row)
        db.insert_sections(
            title,
            [
                (
                    position,
                    section_title,
                    content,
                    str(article_dir / f"{position:02d}-{safe_filename(section_title)}.webp"),
                )
                for position, (section_title, content) in enumerate(fetched, start=1)
            ],
        )
        sections = db.get_sections(title)

    for section in sections:
        output_path = Path(section["output_path"])
        if section["status"] == "done" and (
            not r2
            or (
                section["r2_key"]
                and section["r2_bucket"] == r2.bucket
            )
        ):
            continue

        # The image may have been saved right before an interruption.
        if not output_path.exists():
            if section["status"] == "done":
                raise FileNotFoundError(
                    f"Completed infographic is missing locally: {output_path}; "
                    "cannot upload it to R2"
                )
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
            partial = output_path.with_suffix(".partial")
            generate_infographic_image(gemini, prompt, partial)
            partial.replace(output_path)

        ensure_webp(output_path)
        if r2:
            key = infographic_key(row["pageid"], section["position"], output_path)
            r2.put_image(key, output_path.read_bytes(), "image/webp")
            db.mark_section_uploaded(title, section["position"], r2.bucket, key)

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
        reel_key = f"reels/{row['pageid']}.json"
        r2.put_json(reel_key, reel)
        db.update_article(
            title,
            r2_reel_bucket=r2.bucket,
            r2_reel_key=reel_key,
            r2_reel_uploaded_at=utc_now_iso(),
        )
    else:
        reel_path = topic_dir(args.output_dir, row) / "reel.json"
        reel_path.parent.mkdir(parents=True, exist_ok=True)
        reel_path.write_bytes(to_json_bytes(reel))
    db.update_article(title, status="done", error=None)
    return "done"


def backfill_labels(
    db: ProgressDB,
    wiki: WikipediaClient,
    r2: R2Store | None,
    level: int,
    output_dir: Path,
) -> int:
    """
    Add vital_field / vital_subfield to reels that are already finished.

    Only Wikipedia category listings are fetched; reel JSON is rebuilt from the
    progress DB, so no Gemini, image or embedding work is repeated. New articles
    found in the categories are not added. Returns the number of reels rewritten.
    """
    for vital_level in range(1, level + 1):
        titles = wiki.discover_vital_titles(vital_level)
        db.save_labels(titles)
        logging.info("Labelled level-%d articles (%d listed)", vital_level, len(titles))

    rewritten = 0
    for row in db.finished_articles(level):
        reel = make_reel_object(row, db.get_sections(row["title"]))
        if r2 and row["r2_reel_key"]:
            r2.put_json(row["r2_reel_key"], reel)
        else:
            reel_path = topic_dir(output_dir, row) / "reel.json"
            if not reel_path.exists():
                continue
            reel_path.write_bytes(to_json_bytes(reel))
        rewritten += 1
    db.reset_vectorized(level)
    return rewritten
