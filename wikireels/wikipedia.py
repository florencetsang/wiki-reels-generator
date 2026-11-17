"""Wikipedia API: vital-article discovery and article metadata."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from .config import USER_AGENT


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_DELAY_SECONDS = 0.10

ARTICLE_NS = 0
TALK_NS = 1
CATEGORY_NS = 14


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
