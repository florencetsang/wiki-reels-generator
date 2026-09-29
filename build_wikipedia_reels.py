#!/usr/bin/env python3
"""
Build Wikipedia reel objects for Level 3-5 Vital Articles.

Stored in Cloudflare R2:
  reels/<pageid>.json
  reels/<pageid>/infographics/001.png
  reels/<pageid>/infographics/002.png
  ...
  manifests/<run_id>.json

Usage:
  pip install -r requirements.txt
  cp .env.example .env
  # Fill in .env values
  python build_wikipedia_reels.py --dry-run --limit 3
  python build_wikipedia_reels.py --limit 25
  python build_wikipedia_reels.py --resume
  python build_wikipedia_reels.py --workers 1

Notes:
- `--resume` skips an article when reels/<pageid>.json is already present in R2.
- Embeddings are cached locally in .cache/embeddings/.
- Section count uses top-level article sections only (level "2"), excluding
  See also, References, Further reading, External links, Notes, Bibliography,
  Sources, and Citations.
- One Gemini image call is made per retained section.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import boto3
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_dotenv()

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"

USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "WikipediaReelsBot/1.0 (contact@example.com)",
)

GEMINI_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

R2_BUCKET = os.getenv("R2_BUCKET", "")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv(
    "R2_ENDPOINT_URL",
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    if R2_ACCOUNT_ID
    else "",
)

CACHE_DIR = Path(".cache")
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# These three root categories collectively represent the requested data set.
VITAL_ROOT_CATEGORIES = [
    "Category:Wikipedia level-3 vital articles",
    "Category:Wikipedia level-4 vital articles",
    "Category:Wikipedia level-5 vital articles",
]

# Do not generate pages for these conventional reference/navigation sections.
EXCLUDED_SECTION_TITLES = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
}

# Namespace 0 is normal encyclopedia articles.
ARTICLE_NAMESPACE = 0

# Conservative throttling. Increase only after testing and respecting Wikimedia.
WIKIPEDIA_DELAY_SECONDS = 0.10
GEMINI_DELAY_SECONDS = 0.25


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^\w.-]+", "-", value, flags=re.UNICODE)
    return value[:160].strip("-") or "untitled"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# -----------------------------------------------------------------------------
# Wikipedia client
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

    def discover_articles_from_category_tree(
        self,
        root_categories: list[str],
    ) -> dict[int, dict[str, Any]]:
        """
        Recursively traverse category members.

        Returns a de-duplicated mapping keyed by page ID:
          {
            pageid: {
              "title": "...",
              "vital_levels": [3, 4, 5]
            }
          }

        A title may be in more than one Vital level category, so levels are
        preserved as metadata while page IDs remain unique.
        """
        found: dict[int, dict[str, Any]] = {}

        for root in root_categories:
            level_match = re.search(r"level-(\d+)", root, re.IGNORECASE)
            level = int(level_match.group(1)) if level_match else None

            logging.info("Discovering category tree: %s", root)

            queue = [root]
            seen_categories: set[str] = set()

            while queue:
                category = queue.pop(0)

                if category in seen_categories:
                    continue
                seen_categories.add(category)

                cmcontinue: str | None = None

                while True:
                    result = self.api(
                        {
                            "action": "query",
                            "list": "categorymembers",
                            "cmtitle": category,
                            "cmtype": "page|subcat",
                            "cmlimit": "max",
                            "cmprop": "ids|title|type",
                            **({"cmcontinue": cmcontinue} if cmcontinue else {}),
                        }
                    )

                    for member in result.get("query", {}).get(
                        "categorymembers",
                        [],
                    ):
                        member_type = member.get("type")
                        title = member["title"]

                        if member_type == "subcat":
                            queue.append(title)
                            continue

                        if member.get("ns") != ARTICLE_NAMESPACE:
                            continue

                        pageid = int(member["pageid"])
                        entry = found.setdefault(
                            pageid,
                            {
                                "pageid": pageid,
                                "title": title,
                                "vital_levels": [],
                            },
                        )

                        if level is not None and level not in entry["vital_levels"]:
                            entry["vital_levels"].append(level)

                    cmcontinue = result.get("continue", {}).get("cmcontinue")
                    if not cmcontinue:
                        break

        for item in found.values():
            item["vital_levels"].sort()

        return found

    def get_article_metadata(self, title: str) -> dict[str, Any] | None:
        """
        Fetches title, pageid, lead extract, categories, canonical URL,
        last revision ID, and timestamp.

        `exintro=1` means the summary is the article lead rather than an
        LLM-generated summary.
        """
        result = self.api(
            {
                "action": "query",
                "redirects": "1",
                "prop": "extracts|categories|info|revisions",
                "titles": title,
                "exintro": "1",
                "explaintext": "1",
                "cllimit": "max",
                "inprop": "url",
                "rvprop": "ids|timestamp",
                "rvlimit": "1",
            }
        )

        pages = result.get("query", {}).get("pages", [])
        if not pages:
            return None

        page = pages[0]
        if page.get("missing") or page.get("invalid"):
            return None

        categories = []
        for category in page.get("categories", []):
            category_title = category.get("title", "")
            if category_title.startswith("Category:"):
                categories.append(category_title.removeprefix("Category:"))

        revisions = page.get("revisions", [])
        revision = revisions[0] if revisions else {}

        return {
            "pageid": int(page["pageid"]),
            "title": page["title"],
            "summary": (page.get("extract") or "").strip(),
            "categories": sorted(set(categories)),
            "canonical_url": page.get("canonicalurl")
            or f"https://en.wikipedia.org/wiki/{quote(page['title'].replace(' ', '_'))}",
            "revision_id": revision.get("revid"),
            "revision_timestamp": revision.get("timestamp"),
        }

    def get_eligible_sections(self, title: str) -> list[str]:
        """
        Returns top-level article section names only (MediaWiki level "2").
        """
        result = self.api(
            {
                "action": "parse",
                "page": title,
                "prop": "sections",
                "redirects": "1",
            }
        )

        all_sections = result.get("parse", {}).get("sections", [])
        retained: list[str] = []

        for section in all_sections:
            # Level 2 corresponds to standard major headings in article prose.
            if str(section.get("level")) != "2":
                continue

            line = re.sub(r"\s+", " ", section.get("line", "")).strip()
            normalized = line.casefold()

            if not line or normalized in EXCLUDED_SECTION_TITLES:
                continue

            retained.append(line)

        return retained


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

class LocalEmbeddingService:
    def __init__(self, model_name: str) -> None:
        logging.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_once(self, summary: str, categories: list[str]) -> list[float]:
        """
        Caches the embedding by model name and source-content hash.

        Embeddings are normalized; cosine similarity can be calculated using
        a dot product if your future vector store supports it.
        """
        embedding_source = (
            f"Summary:\n{summary}\n\n"
            f"Wikipedia categories:\n"
            + "\n".join(f"- {category}" for category in categories)
        )

        cache_key = sha256_text(
            f"model={self.model_name}\ncontent={embedding_source}"
        )
        cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}.json"

        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        vector = self.model.encode(
            embedding_source,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = [float(item) for item in vector.tolist()]
        cache_path.write_text(
            json.dumps(result, separators=(",", ":")),
            encoding="utf-8",
        )
        return result


# -----------------------------------------------------------------------------
# Gemini / Nano Banana image generation
# -----------------------------------------------------------------------------

class GeminiInfographicService:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required.")

        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def infographic_prompt(
        article_title: str,
        all_sections: list[str],
        page_number: int,
        section_title: str,
    ) -> str:
        """
        Keeps the user's requested prompt language, while adding explicit
        instructions that ensure a single returned asset is page i of n.
        """
        n = len(all_sections)
        article_url = (
            "https://en.wikipedia.org/wiki/"
            + quote(article_title.replace(" ", "_"))
        )
        sections_text = ", ".join(all_sections)

        return f"""
Generate {n} infographics about the {sections_text} of {article_title}: {article_url}

Generate infographic page {page_number} of {n}. This page must focus primarily
on the section "{section_title}".

Create a polished, editorial, fact-oriented vertical infographic suitable for a
short-form video reel. Use a 9:16 composition. Include a clear title,
well-structured visual hierarchy, icons, diagrams, timelines, maps, or charts
when useful. Avoid fabricating precise statistics, dates, quotations, citations,
or claims not safely supported by the cited Wikipedia article. Keep text concise,
legible, and in English. Do not add a watermark or logo.
""".strip()

    def generate_page(
        self,
        article_title: str,
        all_sections: list[str],
        page_number: int,
        section_title: str,
    ) -> tuple[bytes, str]:
        prompt = self.infographic_prompt(
            article_title=article_title,
            all_sections=all_sections,
            page_number=page_number,
            section_title=section_title,
        )

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                response_format={
                    "image": {
                        "aspect_ratio": "9:16",
                    }
                },
            ),
        )

        for part in response.parts or []:
            if part.inline_data and part.inline_data.data:
                data = part.inline_data.data
                mime_type = part.inline_data.mime_type or "image/png"

                # SDK versions can expose bytes or a base64 string.
                if isinstance(data, str):
                    data = base64.b64decode(data)

                return data, mime_type

        raise RuntimeError(
            f"Gemini returned no image for {article_title!r}, "
            f"section {section_title!r}."
        )


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
            Body=stable_json(value),
            ContentType="application/json; charset=utf-8",
        )

    def put_image(self, key: str, content: bytes, mime_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=mime_type,
        )


# -----------------------------------------------------------------------------
# Reel assembly
# -----------------------------------------------------------------------------

def make_reel_object(
    metadata: dict[str, Any],
    vital_levels: list[int],
    eligible_sections: list[str],
    infographic_items: list[dict[str, Any]],
    embedding: list[float],
) -> dict[str, Any]:
    """
    The storage object intentionally retains source URL/revision metadata.
    This is useful for attribution, refresh jobs, auditability, and detecting
    articles whose Wikipedia source has changed.
    """
    return {
        "schema_version": "1.0",
        "type": "wikipedia_reel",
        "id": f"wikipedia:{metadata['pageid']}",
        "title": metadata["title"],
        "summary": metadata["summary"],
        "category_tags": metadata["categories"],
        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "dimensions": len(embedding),
            "normalized": True,
            "vector": embedding,
        },
        "infographics": {
            "page_count": len(eligible_sections),
            "sections": eligible_sections,
            "assets": infographic_items,
            "generator": {
                "provider": "Google Gemini",
                "model": GEMINI_MODEL,
                "aspect_ratio": "9:16",
            },
        },
        "source": {
            "project": "English Wikipedia",
            "article_url": metadata["canonical_url"],
            "pageid": metadata["pageid"],
            "revision_id": metadata["revision_id"],
            "revision_timestamp": metadata["revision_timestamp"],
            "vital_article_levels": vital_levels,
            "license_note": (
                "Wikipedia text is generally available under "
                "CC BY-SA 4.0; preserve attribution and verify applicable "
                "source licensing before redistribution."
            ),
        },
        "generated_at": utc_now_iso(),
    }


def process_one_article(
    wiki: WikipediaClient,
    embeddings: LocalEmbeddingService,
    gemini: GeminiInfographicService | None,
    r2: R2Store | None,
    discovered: dict[str, Any],
    dry_run: bool,
    resume: bool,
    generate_images: bool,
) -> dict[str, Any]:
    title = discovered["title"]
    original_pageid = int(discovered["pageid"])
    eventual_reel_key = f"reels/{original_pageid}.json"

    if resume and r2 and r2.object_exists(eventual_reel_key):
        return {
            "title": title,
            "pageid": original_pageid,
            "status": "skipped_existing",
        }

    metadata = wiki.get_article_metadata(title)
    if metadata is None:
        return {
            "title": title,
            "pageid": original_pageid,
            "status": "skipped_missing",
        }

    # Redirects can resolve to a new page ID.
    pageid = metadata["pageid"]
    reel_key = f"reels/{pageid}.json"

    if resume and r2 and r2.object_exists(reel_key):
        return {
            "title": metadata["title"],
            "pageid": pageid,
            "status": "skipped_existing",
        }

    summary = metadata["summary"]
    if not summary:
        return {
            "title": metadata["title"],
            "pageid": pageid,
            "status": "skipped_no_summary",
        }

    sections = wiki.get_eligible_sections(metadata["title"])
    embedding = embeddings.embed_once(summary, metadata["categories"])

    infographic_items: list[dict[str, Any]] = []

    if generate_images:
        if gemini is None:
            raise RuntimeError("Gemini service is required for image generation.")

        for page_number, section_title in enumerate(sections, start=1):
            key = f"reels/{pageid}/infographics/{page_number:03d}.png"

            if resume and r2 and r2.object_exists(key):
                infographic_items.append(
                    {
                        "page": page_number,
                        "section": section_title,
                        "r2_key": key,
                        "mime_type": "image/png",
                        "status": "reused",
                    }
                )
                continue

            image_bytes, mime_type = gemini.generate_page(
                article_title=metadata["title"],
                all_sections=sections,
                page_number=page_number,
                section_title=section_title,
            )

            if not dry_run and r2:
                extension = "png" if mime_type == "image/png" else "webp"
                key = (
                    f"reels/{pageid}/infographics/"
                    f"{page_number:03d}.{extension}"
                )
                r2.put_image(key, image_bytes, mime_type)

            infographic_items.append(
                {
                    "page": page_number,
                    "section": section_title,
                    "r2_key": key,
                    "mime_type": mime_type,
                    "status": "generated" if not dry_run else "dry_run",
                }
            )

            time.sleep(GEMINI_DELAY_SECONDS)

    reel = make_reel_object(
        metadata=metadata,
        vital_levels=discovered["vital_levels"],
        eligible_sections=sections,
        infographic_items=infographic_items,
        embedding=embedding,
    )

    if not dry_run and r2:
        r2.put_json(reel_key, reel)

    return {
        "title": metadata["title"],
        "pageid": pageid,
        "status": "completed" if not dry_run else "dry_run",
        "reel_key": reel_key,
        "section_count": len(sections),
        "infographic_count": len(infographic_items),
    }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of unique articles to process. 0 = all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch Wikipedia and create embeddings, but do not upload to R2.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip reels/assets that already exist in R2.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Create reel metadata and embeddings without Gemini calls.",
    )
    parser.add_argument(
        "--titles-file",
        type=Path,
        help=(
            "Optional newline-delimited article titles. When specified, "
            "bypasses Vital Article category discovery."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_titles_file(path: Path) -> dict[int, dict[str, Any]]:
    """
    Assigns temporary IDs for titles-file mode. Actual Wikipedia page IDs are
    resolved before the reel is stored.
    """
    records: dict[int, dict[str, Any]] = {}

    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        title = raw.strip()
        if not title or title.startswith("#"):
            continue

        records[-index] = {
            "pageid": -index,
            "title": title,
            "vital_levels": [],
        }

    return records


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.dry_run and not R2_BUCKET:
        logging.error("R2 configuration is required unless --dry-run is used.")
        return 2

    wiki = WikipediaClient()
    embeddings = LocalEmbeddingService(EMBEDDING_MODEL_NAME)
    r2 = None if args.dry_run else R2Store()

    generate_images = not args.no_images
    gemini = GeminiInfographicService() if generate_images else None

    if args.titles_file:
        discovered = load_titles_file(args.titles_file)
    else:
        discovered = wiki.discover_articles_from_category_tree(
            VITAL_ROOT_CATEGORIES
        )

    articles = sorted(
        discovered.values(),
        key=lambda item: item["title"].casefold(),
    )

    if args.limit > 0:
        articles = articles[: args.limit]

    logging.info("Unique articles selected: %d", len(articles))
    logging.info("Image generation enabled: %s", generate_images)
    logging.info("Dry run: %s", args.dry_run)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict[str, Any]] = []

    for index, item in enumerate(articles, start=1):
        logging.info(
            "[%d/%d] Processing %s",
            index,
            len(articles),
            item["title"],
        )

        try:
            result = process_one_article(
                wiki=wiki,
                embeddings=embeddings,
                gemini=gemini,
                r2=r2,
                discovered=item,
                dry_run=args.dry_run,
                resume=args.resume,
                generate_images=generate_images,
            )
            results.append(result)
            logging.info("Result: %s", result["status"])

        except Exception as exc:
            logging.exception("Failed: %s", item["title"])
            results.append(
                {
                    "title": item["title"],
                    "pageid": item["pageid"],
                    "status": "failed",
                    "error": str(exc),
                }
            )

    manifest = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "settings": {
            "gemini_model": GEMINI_MODEL,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "generate_images": generate_images,
            "dry_run": args.dry_run,
            "resume": args.resume,
        },
        "summary": {
            "selected": len(articles),
            "completed": sum(
                result["status"] in {"completed", "dry_run"}
                for result in results
            ),
            "failed": sum(
                result["status"] == "failed"
                for result in results
            ),
            "skipped": sum(
                result["status"].startswith("skipped")
                for result in results
            ),
        },
        "results": results,
    }

    local_manifest = Path(f"manifest-{run_id}.json")
    local_manifest.write_bytes(stable_json(manifest))

    if not args.dry_run and r2:
        r2.put_json(f"manifests/{run_id}.json", manifest)

    logging.info("Manifest written locally: %s", local_manifest)
    logging.info("Run summary: %s", manifest["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())