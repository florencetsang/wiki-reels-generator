"""Wikipedia API: vital-article discovery and article metadata."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, NamedTuple

import requests

from .config import USER_AGENT


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_DELAY_SECONDS = 0.10

ARTICLE_NS = 0
TALK_NS = 1
CATEGORY_NS = 14

# Top-level Vital Articles fields, keyed by the stable slug the app uses.
# Values are the (case-insensitive) names used in
# "Category:Wikipedia level-N vital articles in <name>".
VITAL_FIELDS: dict[str, tuple[str, ...]] = {
    "people": ("people",),
    "history": ("history",),
    "geography": ("geography",),
    "arts": ("arts",),
    "philosophy_religion": ("philosophy and religion",),
    "everyday_life": ("everyday life",),
    "society": ("society and social sciences",),
    "biology_health": ("biological and health sciences", "biology and health sciences"),
    "physical_sciences": ("physical sciences",),
    "technology": ("technology",),
    "mathematics": ("mathematics",),
}
_FIELD_BY_NAME = {
    name: slug for slug, names in VITAL_FIELDS.items() for name in names
}
_CATEGORY_TOPIC = re.compile(r"vital articles in (.+)$", re.IGNORECASE)


class VitalTopic(NamedTuple):
    field: str | None
    subfield: str | None


def category_topic(category: str, parent: VitalTopic) -> VitalTopic:
    """
    Topic of a vital-article category, e.g.
    "Category:Wikipedia level-4 vital articles in Physical sciences" -> physical_sciences.
    Deeper "... in <name>" categories that are not top-level fields become the
    subfield of the field inherited from their parent category.
    """
    match = _CATEGORY_TOPIC.search(category)
    if not match:
        return parent
    name = match.group(1).strip()
    field = _FIELD_BY_NAME.get(name.casefold())
    if field:
        return VitalTopic(field, None)
    return VitalTopic(parent.field, name) if parent.field else parent


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

    def discover_vital_titles(self, level: int) -> dict[str, VitalTopic]:
        """
        Vital-article categories are populated by talk-page banners, so members
        are mostly "Talk:<Article>" pages. Subcategories are followed only when
        they belong to the same level. Each title is mapped to the field (and
        subfield, when nested deeper) of the category it was found in.
        """
        level_marker = f"level-{level} vital"
        root = VitalTopic(None, None)
        queue = [(f"Category:Wikipedia level-{level} vital articles", root)]
        seen: set[str] = set()
        titles: dict[str, VitalTopic] = {}

        def add(title: str, topic: VitalTopic) -> None:
            # Prefer the most specific topic when a title is in several categories.
            current = titles.get(title)
            if current is None or (topic.field and not current.field) or (
                topic.field == current.field and topic.subfield and not current.subfield
            ):
                titles[title] = topic

        while queue:
            category, parent = queue.pop()
            if category in seen:
                continue
            seen.add(category)
            topic = category_topic(category, parent)
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
                            queue.append((title, topic))
                    elif ns == TALK_NS:
                        add(title.removeprefix("Talk:"), topic)
                    elif ns == ARTICLE_NS:
                        add(title, topic)

                if "continue" not in result:
                    break
                continuation = result["continue"]

        return dict(sorted(titles.items(), key=lambda item: item[0].casefold()))

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
