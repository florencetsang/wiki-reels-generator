```markdown
# Wikipedia Reels Builder

Build structured “reel” objects from English Wikipedia Vital Articles (Levels 3–5).

For each article, the script:

1. Discovers articles from the Level 3, 4, and 5 Vital Articles category trees—or reads a supplied title list.
2. Fetches article metadata, lead text, categories, revision information, and top-level sections.
3. Creates a local semantic embedding from the article lead and category tags.
4. Generates one vertical infographic per eligible top-level section using Google Gemini.
5. Stores reel metadata, infographic assets, and a run manifest in Cloudflare R2.

## Output layout

Objects are written to Cloudflare R2 in this structure:

```text
reels/<pageid>.json
reels/<pageid>/infographics/001.png
reels/<pageid>/infographics/002.png
...
manifests/<run_id>.json
```

A local copy of every run manifest is also created:

```text
manifest-YYYYMMDDTHHMMSSZ.json
```

## Requirements

- Python 3.10 or newer
- A Google Gemini API key with access to the configured image-generation model
- A Cloudflare R2 bucket and S3-compatible API credentials
- Internet access to Wikipedia, Hugging Face model downloads, Google Gemini, and Cloudflare R2

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Depending on your environment, `sentence-transformers` may install PyTorch automatically. If it does not, install the appropriate PyTorch build for your platform before running the script.

## Configuration

Copy the example environment file and set the values in `.env`:

```bash
cp .env.example .env
```

## Usage

### Test a small batch without uploading

```bash
python build_wikipedia_reels.py --dry-run --limit 3
```

This still fetches Wikipedia content, generates embeddings, and—unless `--no-images` is supplied—makes Gemini image-generation requests. It only prevents uploads to R2. Instead, reel objects and generated infographic assets are written locally under `reels/`, mirroring the R2 key layout:

```text
reels/<pageid>.json
reels/<pageid>/infographics/001.png
reels/<pageid>/infographics/002.png
...
```

With `--dry-run --resume`, existing local files are reused instead of calling Gemini again.

To test Wikipedia retrieval and embeddings **without** Gemini calls:

```bash
python build_wikipedia_reels.py --dry-run --no-images --limit 3
```

### Process Vital Articles

Process the first 25 articles alphabetically:

```bash
python build_wikipedia_reels.py --limit 25
```

Process all discovered Level 3–5 Vital Articles:

```bash
python build_wikipedia_reels.py
```

### Resume an interrupted run

```bash
python build_wikipedia_reels.py --resume
```

When `--resume` is set, the script skips an article if its reel JSON already exists in R2. It also reuses an existing infographic asset when the expected R2 object exists.

### Build metadata and embeddings only

```bash
python build_wikipedia_reels.py --no-images --limit 25
```

This writes reel JSON without generating infographic assets.

### Supply specific Wikipedia article titles

Create a newline-delimited text file:

```text
Ada Lovelace
Apollo program
Climate change
# Lines beginning with # are ignored
```

Run:

```bash
python build_wikipedia_reels.py --titles-file titles.txt
```

You can combine this with other options:

```bash
python build_wikipedia_reels.py \
  --titles-file titles.txt \
  --no-images \
  --resume
```

### Enable detailed logging

```bash
python build_wikipedia_reels.py --limit 3 --log-level DEBUG
```

Available levels:

```text
DEBUG
INFO
WARNING
ERROR
```

## Command-line options

| Option | Description |
|---|---|
| `--limit N` | Process at most `N` unique articles. `0` means all selected articles. |
| `--dry-run` | Fetch Wikipedia content and create embeddings, but do not upload objects to R2. Gemini calls still occur unless `--no-images` is used. |
| `--resume` | Skip reel JSON and infographic assets that are already present in R2. |
| `--no-images` | Skip Gemini image generation and create only metadata plus embeddings. |
| `--titles-file PATH` | Read article titles from a newline-delimited file instead of discovering Vital Articles. |
| `--log-level LEVEL` | Set logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

## Reel JSON format

Each reel is stored as `reels/<pageid>.json` and resembles:

```json
{
  "schema_version": "1.0",
  "type": "wikipedia_reel",
  "id": "wikipedia:12345",
  "title": "Example article",
  "summary": "The article lead text.",
  "category_tags": ["Example categories"],
  "embedding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "normalized": true,
    "vector": []
  },
  "infographics": {
    "page_count": 2,
    "sections": ["History", "Legacy"],
    "assets": [
      {
        "page": 1,
        "section": "History",
        "r2_key": "reels/12345/infographics/001.png",
        "mime_type": "image/png",
        "status": "generated"
      }
    ],
    "generator": {
      "provider": "Google Gemini",
      "model": "gemini-2.5-flash-image",
      "aspect_ratio": "9:16"
    }
  },
  "source": {
    "project": "English Wikipedia",
    "article_url": "https://en.wikipedia.org/wiki/Example_article",
    "pageid": 12345,
    "revision_id": 123456789,
    "revision_timestamp": "2026-01-01T00:00:00Z",
    "vital_article_levels": [3],
    "license_note": "..."
  },
  "generated_at": "2026-01-01T00:00:00+00:00"
}
```

## Section selection

The script creates one infographic for each retained top-level Wikipedia section.

It includes only MediaWiki level-2 headings and excludes conventional navigation/reference headings:

- See also
- References
- External links
- Further reading
- Notes
- Bibliography
- Sources
- Citations

Articles with no eligible sections can still produce a reel JSON object, but their infographic list will be empty.

## Caching

Embedding vectors are cached locally under:

```text
.cache/embeddings/
```

The cache key is derived from:

- The configured embedding model name
- The article lead text
- The article category list

Delete `.cache/embeddings/` to force regeneration of all embeddings.

## Important operational notes

- **Gemini cost:** one Gemini image-generation request is made for every retained article section when images are enabled.
- **Dry runs can incur cost:** `--dry-run` prevents R2 uploads but does not suppress Gemini calls. Add `--no-images` when testing without image-generation cost.
- **Wikipedia etiquette:** configure a descriptive `WIKIPEDIA_USER_AGENT` with a valid contact method and keep request rates conservative.
- **R2 permissions:** the R2 credentials need permission to read object metadata and write objects in the configured bucket.
- **Model downloads:** the sentence-transformer model may be downloaded on first run.
- **Image formats:** generated images are stored as `.png` when Gemini returns `image/png`; otherwise they are stored as `.webp`.

## Troubleshooting

### `GEMINI_API_KEY is required.`

Set `GEMINI_API_KEY` in `.env`, then rerun the command.

If using `--no-images`, Gemini is not initialized and a Gemini key is not required.

### Missing R2 configuration

Non-dry runs require:

```text
R2_BUCKET
R2_ENDPOINT_URL
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
```

Use `--dry-run` to test without R2 configuration.

### R2 endpoint errors

Confirm that `R2_ENDPOINT_URL` uses the account-specific Cloudflare R2 endpoint:

```text
https://<account-id>.r2.cloudflarestorage.com
```

Also verify that the access key and secret are R2 S3 API credentials, not a general Cloudflare API token.

## Attribution and licensing

Wikipedia article text is generally available under the Creative Commons Attribution-ShareAlike 4.0 license. Reel objects retain article URLs, page IDs, revision IDs, and revision timestamps to support attribution, auditing, and refresh workflows.

Before redistributing generated content, verify the licensing requirements for:

- Wikipedia source text and media
- Any source images or media incorporated by upstream services
- Generated image assets
- Your downstream product and distribution model