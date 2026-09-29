import argparse
import os
import re
from pathlib import Path

import wikipediaapi
from google import genai
from google.genai import errors, types


TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
SKIPPED_SECTIONS = {
    "See also",
    "Notes",
    "Footnotes",
    "References",
    "Citations",
    "Bibliography",
    "Sources",
    "Works cited",
    "Further reading",
    "External links",
}


def create_client() -> genai.Client:
    """Create a Gemini client from the GEMINI_API_KEY environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before running this program.")
    return genai.Client(api_key=api_key)


def section_text(section) -> str:
    """Combine a top-level section with the text of all its subsections."""
    parts = [section.text]
    parts.extend(section_text(child) for child in section.sections)
    return "\n\n".join(part.strip() for part in parts if part.strip())


def fetch_wikipedia_sections(title: str) -> list[tuple[str, str]]:
    """Fetch meaningful top-level sections from a Wikipedia page."""
    user_agent = os.getenv(
        "WIKIPEDIA_USER_AGENT",
        "WikiToInfographicBot/1.0 (contact@example.com)",
    )
    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")
    page = wiki.page(title)

    if not page.exists():
        raise ValueError(f"Wikipedia page '{title}' does not exist.")

    sections = []
    for section in page.sections:
        text = section_text(section)
        if section.title not in SKIPPED_SECTIONS and text:
            sections.append((section.title, text[:5000]))

    if not sections:
        raise ValueError(f"Wikipedia page '{title}' has no usable sections.")
    return sections


def generate_infographic_prompt(
    client: genai.Client,
    article_title: str,
    section_title: str,
    section_content: str,
) -> str:
    """Turn one article section into a focused infographic image prompt."""
    prompt = f"""
Create a production-ready image-generation prompt for one standalone infographic.

ARTICLE: {article_title}
SECTION: {section_title}
SOURCE TEXT:
---
{section_content}
---

Use only facts supported by the source text. Select at most five key facts so the
image remains readable. Specify a clear visual hierarchy, relevant imagery or data
visualization, a restrained color palette, and large legible typography. The exact
section title must be the infographic title. Do not mention other article sections.
Return only the image-generation prompt.
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    if not response.text:
        raise RuntimeError(f"No prompt was generated for section '{section_title}'.")
    return response.text.strip()


def generate_infographic_image(
    client: genai.Client,
    prompt: str,
    output_path: Path,
) -> None:
    """Generate one infographic with the configured Gemini image model."""
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="3:4"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    for part in response.parts or []:
        if part.inline_data:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            part.as_image().save(output_path)
            return
    raise RuntimeError(f"The image model returned no image for '{output_path.stem}'.")


def safe_filename(value: str) -> str:
    """Convert a section title to a stable, filesystem-safe filename."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "section"


def wiki_to_infographics(
    wiki_title: str,
    output_directory: Path,
    max_sections: int | None = None,
) -> list[Path]:
    """Generate one PNG infographic for each top-level Wikipedia section."""
    client = create_client()
    sections = fetch_wikipedia_sections(wiki_title)
    if max_sections is not None:
        sections = sections[:max_sections]

    output_paths = []
    print(f"Found {len(sections)} sections in '{wiki_title}'.")
    for index, (section_title, content) in enumerate(sections, start=1):
        output_path = output_directory / (
            f"{index:02d}-{safe_filename(section_title)}.png"
        )
        print(f"[{index}/{len(sections)}] Generating: {section_title}")
        prompt = generate_infographic_prompt(
            client,
            wiki_title,
            section_title,
            content,
        )
        generate_infographic_image(client, prompt, output_path)
        output_paths.append(output_path)
        print(f"Saved: {output_path}")

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one infographic image per Wikipedia section."
    )
    parser.add_argument("title", help="Wikipedia article title")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infographics"),
        help="Directory for generated PNG files (default: infographics)",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        help="Generate only the first N sections (useful for testing API access)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        wiki_to_infographics(
            arguments.title,
            arguments.output_dir,
            arguments.max_sections,
        )
    except errors.ClientError as error:
        if error.code != 403:
            raise
        raise SystemExit(
            "Gemini API access was denied for the project linked to "
            "GEMINI_API_KEY. Create a new API key in an eligible Google AI "
            "Studio project, or contact Google support to restore this "
            "project's access, then update GEMINI_API_KEY."
        ) from None


if __name__ == "__main__":
    main()