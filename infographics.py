#!/usr/bin/env python3
"""
Generate infographics for every Wikipedia Vital Article at a given level.

Per-article pipeline, checkpointed in SQLite (Cloudflare D1 when configured,
otherwise a local file) so a run can be stopped (Ctrl+C, crash, quota
exhaustion) and resumed at any time:
  1. discover  - list level 1..N vital articles from Wikipedia categories
  2. summarize - fetch lead summary, categories, page ID and revision
  3. embed     - compute a local sentence-transformers embedding
  4. images    - one infographic per section, using wikiinfo.py logic
  5. upload    - infographics and reel JSON to Cloudflare R2 when configured:
                   reels/<pageid>.json
                   reels/<pageid>/infographics/001.webp ...
                 otherwise reel JSON is written to <output-dir>/<topic>/reel.json
  6. vectorize - after each run, upsert embeddings of finished reels into a
                 Cloudflare Vectorize index (id = pageid) when configured,
                 otherwise into a local SQLite vector index (--vectors-db)

Code layout (wikireels/):
  config.py     settings from .env
  common.py     shared types and helpers
  progress.py   progress database (local SQLite / D1) and its queries
  wikipedia.py  vital-article discovery and article metadata
  embeddings.py sentence-transformers embeddings
  storage.py    local infographic files and Cloudflare R2
  vectors.py    Vectorize / local vector index and syncing
  pipeline.py   per-article processing
  cli.py        arguments and the run loop

Usage:
  python infographics.py                  # level 4, resumes automatically
  python infographics.py --level 4 --limit 20
  python infographics.py --no-images      # summaries + embeddings only
  python infographics.py --local          # ignore Cloudflare config (or LOCAL_ONLY=true)
  python infographics.py --status         # print progress and exit
  python infographics.py --related Moon   # similar reels from the local vector index
"""

from wikireels.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
