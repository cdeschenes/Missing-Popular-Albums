# Missing Popular Albums

Locate the single most popular album or EP missing from your music library for every artist you already own. The script scans your collection, calls the Last.fm API for playcount-ranked releases, and renders a HTML report with high-res artwork, quick links, and copy-to-clipboard shortcuts.

## Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need a Last.fm API key:

1. Visit https://www.last.fm/api and sign in with your Last.fm account.
2. Choose **Create an API Account**, fill in the short application form, and submit.
3. Copy the generated API key—you'll add it to your `.env` file in the next step.

## Configure Environment

Copy the template and edit values to match your setup (music path, credentials, output locations, etc.).

```bash
cp .env.example .env
```

Open `.env` and provide your `LASTFM_API_KEY` (and any other overrides).

## Run The Script

```bash
python missing_popular_albums.py
```

Optional flags:

- `--limit-artists N` – process only the first `N` artists (useful for testing)
- `--no-cache` – ignore existing Last.fm cache data
- `--workers N` – adjust the thread pool (default matches `.env`)

## HTML Report

Generated at the path defined by `HTML_OUT` (defaults to `missing_popular_albums.html`). It includes:

- Album/artist cards sorted alphabetically
- Last.fm, Discogs, Bandcamp, and YouTube Music links
- Clipboard buttons for pasting `Artist Album` search strings

<p align="center">
  <img src="screenshots/htmloutput.png" alt="Missing Popular Albums screenshot" width="640">
</p>

## Logging

Execution details are written to `missing_popular_albums.log` (path controlled by `.env`). Logs include scan counts, cache hits/misses, API errors, and skipped or missing albums. The file rotates automatically (1 MB, 3 backups).

## Room for Improvement

- Push/pull integration with other catalog services (e.g., MusicBrainz)
- Smarter album filtering (recognize studio releases without API tag lookups)
- Export to CSV

---

This was done with assitive coding with OpenAI GPT CODEX
