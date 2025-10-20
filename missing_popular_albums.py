"""Identify the single most popular missing album per artist.

The script scans audio files under ``MUSIC_ROOT``, extracts artist/album metadata
from tags or folder names, queries Last.fm for each artist's top albums, and
produces an HTML report listing the highest-playcount Album/EP that is missing
from the local collection. Results are cached and logged for repeat runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import logging
import os
import random
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from mutagen import File
from rapidfuzz import fuzz
from tqdm import tqdm

ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_CONFIG = {
    "MUSIC_ROOT": "/Volumes/NAS/Media/Music/Music_Server",
    "HTML_OUT": "missing_popular_albums.html",
    "CACHE_FILE": ".cache/lastfm_top_albums.json",
    "LOG_FILE": "missing_popular_albums.log",
    "FUZZ_THRESHOLD": "90",
    "DEFAULT_WORKERS": "4",
    "MAX_WORKERS": "8",
    "TOP_ALBUM_LIMIT": "25",
    "TAG_INFO_CHECK_TOP_N": "3",
    "CACHE_VERSION": "2",
    "REQUEST_TIMEOUT": "15",
    "REQUEST_DELAY_MIN": "0.15",
    "REQUEST_DELAY_MAX": "0.3",
    "MAX_RETRIES": "3",
    "LASTFM_API_KEY": "",
}


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to read configuration from {path}: {exc}") from exc
    return data


def load_config() -> Dict[str, str]:
    config = dict(DEFAULT_CONFIG)
    overrides = load_env_file(ENV_FILE)
    for key, value in overrides.items():
        if not value:
            continue
        config[key] = value
    return config


CONFIG = load_config()

MUSIC_ROOT = Path(CONFIG["MUSIC_ROOT"]).expanduser()
HTML_OUT = Path(CONFIG["HTML_OUT"]).expanduser()
CACHE_FILE = Path(CONFIG["CACHE_FILE"]).expanduser()
LOG_FILE = Path(CONFIG["LOG_FILE"]).expanduser()

FUZZ_THRESHOLD = int(CONFIG["FUZZ_THRESHOLD"])
DEFAULT_WORKERS = int(CONFIG["DEFAULT_WORKERS"])
MAX_WORKERS = int(CONFIG["MAX_WORKERS"])

TOP_ALBUM_LIMIT = int(CONFIG["TOP_ALBUM_LIMIT"])
TAG_INFO_CHECK_TOP_N = int(CONFIG["TAG_INFO_CHECK_TOP_N"])
CACHE_VERSION = int(CONFIG["CACHE_VERSION"])
REQUEST_TIMEOUT = float(CONFIG["REQUEST_TIMEOUT"])
REQUEST_DELAY_RANGE = (
    float(CONFIG["REQUEST_DELAY_MIN"]),
    float(CONFIG["REQUEST_DELAY_MAX"]),
)
MAX_RETRIES = int(CONFIG["MAX_RETRIES"])

if REQUEST_DELAY_RANGE[0] > REQUEST_DELAY_RANGE[1]:
    REQUEST_DELAY_RANGE = (REQUEST_DELAY_RANGE[1], REQUEST_DELAY_RANGE[0])

AUDIO_EXTENSIONS = {
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".aiff",
    ".aif",
    ".ogg",
    ".opus",
}

EXCLUDED_ARTIST_KEYWORDS = {
    "various artists",
    "various artist",
    "soundtrack",
    "ost",
    "score",
    "motion picture",
    "original soundtrack",
    "dj mix",
}

EXCLUDED_TITLE_KEYWORDS = {
    "live",
    "compilation",
    "greatest hits",
    "best of",
    "remix",
    "remixes",
    "anthology",
    "collection",
    "expanded",
    "deluxe edition",
    "deluxe",
    "reissue",
    "mixtape",
    "karaoke",
    "instrumental collection",
    "instrumental compilation",
    "soundtrack",
    "single",
}

EXCLUDED_TAGS = {"compilation", "live", "single", "soundtrack"}
ALLOWED_TAGS = {"album", "ep"}

PRIMARY_ARTIST_SPLIT_RE = re.compile(
    r"\s+(?:&|and|feat\.?|featuring|ft\.?|with)\s+",
    flags=re.IGNORECASE,
)

PAREN_PATTERN = re.compile(r"\([^)]*\)")
ALBUM_DIR_PATTERN = re.compile(r"^\s*(\d{4})\s*[-_]\s*(.+)$")


@dataclass
class LocalArtist:
    """Represents an artist discovered in the local library."""

    display_name: str
    normalized_name: str
    albums: Set[str] = field(default_factory=set)
    album_display: Dict[str, str] = field(default_factory=dict)

    def add_album(self, normalized_album: str, display_album: str) -> None:
        if normalized_album not in self.album_display:
            self.album_display[normalized_album] = display_album
        self.albums.add(normalized_album)


@dataclass
class RemoteAlbum:
    """Represents an album returned by Last.fm."""

    title: str
    normalized_title: str
    playcount: int
    image_url: Optional[str]
    url: Optional[str]
    tags: Sequence[str]


@dataclass
class AlbumSuggestion:
    """Represents a suggested missing album for an artist."""

    artist_display: str
    artist_normalized: str
    album_title: str
    album_normalized: str
    image_url: Optional[str]
    lastfm_url: Optional[str]
    playcount: int


class LastFMError(Exception):
    """Custom exception for Last.fm API issues."""


class LastFMClient:
    """Client for interacting with the Last.fm API with retry and politeness handling."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._thread = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread, "session", None)
        if session is None:
            session = requests.Session()
            self._thread.session = session
        return session

    def _get_random(self) -> random.Random:
        rng = getattr(self._thread, "random", None)
        if rng is None:
            seed = time.time_ns() ^ threading.get_ident()
            rng = random.Random(seed)
            self._thread.random = rng
        return rng

    def _request(self, params: Dict[str, str]) -> Dict:
        params_with_key = {**params, "api_key": self.api_key, "format": "json"}
        session = self._get_session()
        rng = self._get_random()
        last_exception: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(rng.uniform(*REQUEST_DELAY_RANGE))
            try:
                response = session.get(
                    "https://ws.audioscrobbler.com/2.0/",
                    params=params_with_key,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 429:
                    raise LastFMError("Rate limited by Last.fm")
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise LastFMError(payload.get("message", "Unknown Last.fm error"))
                return payload
            except (requests.RequestException, ValueError, LastFMError) as exc:
                last_exception = exc
                if attempt == MAX_RETRIES:
                    break
                backoff = (0.5 * (2 ** (attempt - 1))) + rng.uniform(0, 0.25)
                logging.debug(
                    "Retrying Last.fm request (%s/%s) after %.2fs due to: %s",
                    attempt,
                    MAX_RETRIES,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
        raise LastFMError(str(last_exception or "Unknown Last.fm error"))

    def artist_top_albums(self, artist_name: str, limit: int = TOP_ALBUM_LIMIT) -> Dict:
        return self._request(
            {
                "method": "artist.getTopAlbums",
                "artist": artist_name,
                "autocorrect": "1",
                "limit": str(limit),
            }
        )

    def album_info(self, artist_name: str, album_title: str) -> Optional[Dict]:
        try:
            return self._request(
                {
                    "method": "album.getInfo",
                    "artist": artist_name,
                    "album": album_title,
                    "autocorrect": "1",
                }
            )
        except LastFMError as exc:
            logging.debug(
                "album.getInfo failed for %s - %s: %s", artist_name, album_title, exc
            )
            return None


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handlers = [
        file_handler,
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    logging.getLogger("mutagen").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.WARNING)


def normalize_diacritics(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
    )


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = normalize_diacritics(value)
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = normalize_spaces(value)
    if value.startswith("the "):
        value = value[4:]
    return normalize_spaces(value)


def normalize_album_title(value: str) -> str:
    if not value:
        return ""
    value = PAREN_PATTERN.sub(" ", value)
    value = normalize_text(value)
    replacements = [
        "deluxe edition",
        "deluxe",
        "expanded edition",
        "expanded",
        "remaster",
        "remastered",
        "special edition",
        "limited edition",
        "bonus track version",
        "anniversary edition",
        "20th anniversary",
        "30th anniversary",
        "40th anniversary",
    ]
    for keyword in replacements:
        value = value.replace(keyword, " ")
    return normalize_spaces(value)


def primary_artist_name(name: str) -> str:
    if not name:
        return ""
    return PRIMARY_ARTIST_SPLIT_RE.split(name)[0].strip()


def is_artist_excluded(normalized_artist: str) -> bool:
    return any(keyword in normalized_artist for keyword in EXCLUDED_ARTIST_KEYWORDS)


def extract_tag_value(tags: Dict[str, Iterable[str]], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = tags.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            return value[0]
        return value
    return None


def read_audio_tags(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        audio = File(file_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.debug("Failed to parse tags for %s: %s", file_path, exc)
        return None, None
    if not audio or not audio.tags:
        return None, None
    tags = audio.tags
    artist = extract_tag_value(
        tags,
        (
            "albumartist",
            "album artist",
            "album_artist",
            "artist",
        ),
    )
    album = extract_tag_value(tags, ("album", "albumtitle"))
    return artist, album


def parse_album_from_path(album_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    artist_part = album_dir.parent.name
    album_part = album_dir.name
    match = ALBUM_DIR_PATTERN.match(album_part)
    if match:
        album_part = match.group(2)
    album_part = album_part.replace("_", " ").strip()
    artist_part = artist_part.replace("_", " ").strip()
    return artist_part or None, album_part or None


def add_local_album(
    artists: Dict[str, LocalArtist], artist_name: str, album_name: str, source: str
) -> None:
    normalized_artist = normalize_text(artist_name)
    if not normalized_artist or is_artist_excluded(normalized_artist):
        return
    normalized_album = normalize_album_title(album_name)
    if not normalized_album:
        return
    if normalized_artist not in artists:
        artists[normalized_artist] = LocalArtist(
            display_name=artist_name,
            normalized_name=normalized_artist,
        )
    artist_entry = artists[normalized_artist]
    if len(artist_name) > len(artist_entry.display_name):
        artist_entry.display_name = artist_name
    artist_entry.add_album(normalized_album, album_name)
    logging.debug("Added album '%s' for artist '%s' from %s", album_name, artist_name, source)


def scan_library(root: Path) -> Dict[str, LocalArtist]:
    logging.info("Scanning local library at %s", root)
    artists: Dict[str, LocalArtist] = {}
    if not root.exists():
        logging.error("Music root %s does not exist.", root)
        return artists

    for dirpath, _, filenames in os.walk(root):
        audio_files = [
            Path(dirpath) / filename
            for filename in filenames
            if Path(filename).suffix.lower() in AUDIO_EXTENSIONS
        ]
        if not audio_files:
            continue

        album_dir = Path(dirpath)
        tag_artist: Optional[str] = None
        tag_album: Optional[str] = None
        tag_found = False

        for audio_file in audio_files:
            artist_value, album_value = read_audio_tags(audio_file)
            if artist_value and not tag_artist:
                tag_artist = artist_value
            if album_value and not tag_album:
                tag_album = album_value
            if artist_value or album_value:
                tag_found = True

        if len(audio_files) < 2 and not tag_found:
            continue

        artist_name = tag_artist
        album_name = tag_album

        if not artist_name or not album_name:
            fallback_artist, fallback_album = parse_album_from_path(album_dir)
            artist_name = artist_name or fallback_artist
            album_name = album_name or fallback_album

        if not artist_name or not album_name:
            continue

        add_local_album(artists, artist_name, album_name, source=str(album_dir))

    logging.info("Scan complete. Found %d artists.", len(artists))
    return artists


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != CACHE_VERSION:
            logging.info("Cache version mismatch. Ignoring existing cache.")
            return {}
        return payload.get("artists", {}) or {}
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to load cache from %s: %s", cache_path, exc)
        return {}


def save_cache(cache_path: Path, data: Dict[str, Dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CACHE_VERSION, "artists": data}
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_cached_albums(cache: Dict[str, Dict], artist_key: str) -> Optional[List[RemoteAlbum]]:
    cached_entry = cache.get(artist_key)
    if not cached_entry:
        return None
    albums_data = cached_entry.get("albums")
    if not isinstance(albums_data, list):
        return None
    albums: List[RemoteAlbum] = []
    for album in albums_data:
        try:
            albums.append(
                RemoteAlbum(
                    title=album["title"],
                    normalized_title=album["normalized_title"],
                    playcount=int(album.get("playcount", 0)),
                    image_url=album.get("image_url"),
                    url=album.get("url"),
                    tags=tuple(album.get("tags", [])),
                )
            )
        except (KeyError, ValueError, TypeError):
            logging.debug("Ignoring malformed cache entry for artist key %s", artist_key)
            return None
    return albums


def cache_albums(
    cache: Dict[str, Dict], artist_key: str, artist_name: str, albums: Sequence[RemoteAlbum]
) -> None:
    cache[artist_key] = {
        "artist": artist_name,
        "albums": [
            {
                "title": album.title,
                "normalized_title": album.normalized_title,
                "playcount": album.playcount,
                "image_url": album.image_url,
                "url": album.url,
                "tags": list(album.tags),
            }
            for album in albums
        ],
    }


def upgrade_image_url(url: str) -> str:
    replacements = [
        ("/34s/", "/600x600/"),
        ("/64s/", "/600x600/"),
        ("/128s/", "/600x600/"),
        ("/174s/", "/600x600/"),
        ("/300x300/", "/600x600/"),
        ("/400x400/", "/600x600/"),
    ]
    for source, target in replacements:
        if source in url:
            return url.replace(source, target)
    return url


def extract_image(images: Optional[Sequence[Dict]]) -> Optional[str]:
    if not images:
        return None
    candidates = [img for img in images if isinstance(img, dict)]
    preferred_order = ("mega", "extralarge", "large", "medium", "small")
    for size in preferred_order:
        for image in candidates:
            if image.get("size") == size and image.get("#text"):
                url = image["#text"]
                if not url:
                    continue
                if size in {"mega", "extralarge", "large"}:
                    return upgrade_image_url(url)
                return url
    for image in candidates:
        url = image.get("#text")
        if url:
            return upgrade_image_url(url)
    return None


def fetch_album_tags(client: LastFMClient, artist_name: str, album_title: str) -> Sequence[str]:
    info = client.album_info(artist_name, album_title)
    if not info:
        return ()
    album_section = info.get("album") or {}
    if isinstance(album_section, str):
        return ()
    tags_container = album_section.get("tags") or {}
    if isinstance(tags_container, str):
        return ()
    tags_section = tags_container.get("tag") or []
    if isinstance(tags_section, dict):
        tags_section = [tags_section]
    tag_names = [
        tag.get("name", "")
        for tag in tags_section
        if isinstance(tag, dict) and tag.get("name")
    ]
    return tuple(name.lower() for name in tag_names if name)


def is_album_or_ep(title: str, tags: Sequence[str]) -> bool:
    tags_lower = {tag.lower() for tag in tags}
    if tags_lower & EXCLUDED_TAGS:
        return False
    title_norm = normalize_album_title(title)
    if tags_lower & ALLOWED_TAGS:
        pass
    else:
        for keyword in EXCLUDED_TITLE_KEYWORDS:
            if keyword in title_norm:
                return False
    return bool(title_norm)


def transform_top_albums(
    client: LastFMClient, artist_name: str, albums_payload: Sequence[Dict]
) -> List[RemoteAlbum]:
    remote_albums: List[RemoteAlbum] = []
    seen_titles: Set[str] = set()
    for index, album_data in enumerate(albums_payload):
        title = album_data.get("name")
        if not title:
            continue
        normalized_title = normalize_album_title(title)
        if not normalized_title or normalized_title in seen_titles:
            continue
        playcount_raw = album_data.get("playcount") or 0
        try:
            playcount = int(playcount_raw)
        except (TypeError, ValueError):
            playcount = 0
        image_url = extract_image(album_data.get("image"))
        tags: Sequence[str] = ()
        if index < TAG_INFO_CHECK_TOP_N:
            tags = fetch_album_tags(client, artist_name, title)
        remote_albums.append(
            RemoteAlbum(
                title=title,
                normalized_title=normalized_title,
                playcount=playcount,
                image_url=image_url,
                url=album_data.get("url"),
                tags=tags,
            )
        )
        seen_titles.add(normalized_title)
    remote_albums.sort(key=lambda item: item.playcount, reverse=True)
    return remote_albums


def pick_top_album_ep(albums: Sequence[RemoteAlbum]) -> Optional[RemoteAlbum]:
    for album in albums:
        if is_album_or_ep(album.title, album.tags):
            return album
    return None


def has_album(local_albums: Set[str], candidate_normalized: str) -> bool:
    if candidate_normalized in local_albums:
        return True
    for local_album in local_albums:
        score = fuzz.token_set_ratio(candidate_normalized, local_album)
        if score >= FUZZ_THRESHOLD:
            return True
    return False


def build_card_html(suggestion: AlbumSuggestion) -> str:
    artist_escaped = html.escape(suggestion.artist_display)
    album_escaped = html.escape(suggestion.album_title)
    query = f"{suggestion.artist_display} {suggestion.album_title}"
    query_encoded = requests.utils.quote(query)
    discogs_url = f"https://www.discogs.com/search/?q={query_encoded}&type=release"
    bandcamp_url = f"https://bandcamp.com/search?q={query_encoded}"
    yt_url = f"https://music.youtube.com/search?q={query_encoded}"
    lastfm_url = suggestion.lastfm_url or discogs_url
    if suggestion.image_url:
        cover_html = (
            f'<div class="cover">'
            f'<img src="{html.escape(suggestion.image_url)}" '
            f'alt="{artist_escaped} - {album_escaped} cover art" loading="lazy"></div>'
        )
    else:
        cover_html = '<div class="cover placeholder">No Artwork</div>'
    return (
        "<article class=\"card\">"
        f"{cover_html}"
        "<div class=\"info\">"
        f"<h2>{album_escaped}<span>{artist_escaped}</span></h2>"
        "<div class=\"links\">"
        f"<a href=\"{html.escape(lastfm_url)}\" target=\"_blank\" rel=\"noopener noreferrer\">Last.fm</a>"
        f"<a href=\"{discogs_url}\" target=\"_blank\" rel=\"noopener noreferrer\">Discogs</a>"
        f"<a href=\"{bandcamp_url}\" target=\"_blank\" rel=\"noopener noreferrer\">Bandcamp</a>"
        f"<a href=\"{yt_url}\" target=\"_blank\" rel=\"noopener noreferrer\">YouTube Music</a>"
        "</div>"
        "</div>"
        "</article>"
    )


def render_html(
    suggestions: Sequence[AlbumSuggestion], output_path: Path, total_artists: int
) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    total_suggestions = len(suggestions)
    cards_html = "\n    ".join(build_card_html(item) for item in suggestions)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Missing Popular Albums</title>
  <style>
    :root {{
      color-scheme: dark;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0a0a0a;
      color: #f3f3f3;
    }}
    header {{
      max-width: 1200px;
      margin: 0 auto 2rem;
    }}
    h1 {{
      margin: 0 0 0.5rem;
      font-size: 2.5rem;
      letter-spacing: -0.01em;
    }}
    p.meta {{
      margin: 0;
      color: #a0a0a0;
      font-size: 0.95rem;
    }}
    .grid {{
      display: grid;
      gap: 1.5rem;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      justify-content: center;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      background: #151515;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
      display: flex;
      flex-direction: column;
      transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
      max-width: 300px;
      width: 100%;
      margin: 0 auto;
    }}
    .card:hover {{
      transform: translateY(-6px);
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
    }}
    .cover {{
      background: rgba(255, 255, 255, 0.06);
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 300px;
    }}
    .cover img {{
      width: 100%;
      height: auto;
      display: block;
      object-fit: cover;
    }}
    .cover.placeholder {{
      color: #666;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .info {{
      padding: 1rem 1.2rem 1.4rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }}
    .info h2 {{
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.4;
      font-weight: 600;
    }}
    .info h2 span {{
      display: block;
      color: #7dd6ff;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.25rem;
    }}
    .links {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.5rem;
    }}
    .links a {{
      color: #0b0b0b;
      background: #7dd6ff;
      border-radius: 12px;
      padding: 0.5rem 0.75rem;
      font-weight: 600;
      text-decoration: none;
      transition: background 0.2s ease-in-out;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .links a:hover {{
      background: #54b8e3;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Missing Popular Albums</h1>
    <p class="meta">Generated {timestamp} · {total_suggestions} suggestion(s) across {total_artists} artist(s)</p>
  </header>
  <section class="grid">
    {cards_html}
  </section>
</body>
</html>
"""
    try:
        output_path.write_text(html_content, encoding="utf-8")
    except OSError as exc:
        logging.error("Failed to write HTML output to %s: %s", output_path, exc)
        raise


def summarize_results(
    total_artists: int, suggestions: Sequence[AlbumSuggestion], cache_stats: Dict[str, int]
) -> None:
    logging.info(
        "Finished - %d artist(s) processed, %d suggestion(s) found. Cache hits: %d, misses: %d",
        total_artists,
        len(suggestions),
        cache_stats.get("hits", 0),
        cache_stats.get("misses", 0),
    )
    print(
        f"Processed {total_artists} artist(s). Suggestions: {len(suggestions)}. "
        f"Cache hits: {cache_stats.get('hits', 0)}, misses: {cache_stats.get('misses', 0)}."
    )


def fetch_top_albums_lastfm(
    client: LastFMClient,
    artist_name: str,
    cache: Dict[str, Dict],
    cache_lock: threading.Lock,
    stats_lock: threading.Lock,
    cache_stats: Dict[str, int],
    use_cache: bool,
) -> List[RemoteAlbum]:
    candidate_names = [artist_name]
    simplified = primary_artist_name(artist_name)
    if simplified and simplified.lower() != artist_name.lower():
        candidate_names.append(simplified)
    normalized_keys = [normalize_text(name) for name in candidate_names]

    for candidate, cache_key in zip(candidate_names, normalized_keys):
        if use_cache:
            with cache_lock:
                cached_albums = load_cached_albums(cache, cache_key)
            if cached_albums is not None:
                with stats_lock:
                    cache_stats["hits"] += 1
                logging.debug("Cache hit for artist '%s'", candidate)
                return cached_albums

    for candidate, cache_key in zip(candidate_names, normalized_keys):
        try:
            response = client.artist_top_albums(candidate)
        except LastFMError as exc:
            logging.warning("Failed to fetch top albums for '%s': %s", candidate, exc)
            continue
        top_albums = response.get("topalbums", {}).get("album", [])
        if isinstance(top_albums, dict):
            top_albums = [top_albums]
        remote_albums = transform_top_albums(client, candidate, top_albums)
        if remote_albums:
            with stats_lock:
                cache_stats["misses"] += 1
            with cache_lock:
                cache_albums(cache, cache_key, candidate, remote_albums)
            return remote_albums
    return []


def process_artist(
    artist: LocalArtist,
    client: LastFMClient,
    cache: Dict[str, Dict],
    cache_lock: threading.Lock,
    stats_lock: threading.Lock,
    cache_stats: Dict[str, int],
    use_cache: bool,
) -> Optional[AlbumSuggestion]:
    remote_albums = fetch_top_albums_lastfm(
        client=client,
        artist_name=artist.display_name,
        cache=cache,
        cache_lock=cache_lock,
        stats_lock=stats_lock,
        cache_stats=cache_stats,
        use_cache=use_cache,
    )
    if not remote_albums:
        logging.info("No albums found on Last.fm for %s", artist.display_name)
        return None
    top_album = pick_top_album_ep(remote_albums)
    if not top_album:
        logging.info("No qualifying album/ep for %s", artist.display_name)
        return None
    if has_album(artist.albums, top_album.normalized_title):
        logging.info(
            "Top album for %s already present locally: %s",
            artist.display_name,
            top_album.title,
        )
        return None
    logging.info(
        "Missing album for %s: %s (playcount %d)",
        artist.display_name,
        top_album.title,
        top_album.playcount,
    )
    return AlbumSuggestion(
        artist_display=artist.display_name,
        artist_normalized=artist.normalized_name,
        album_title=top_album.title,
        album_normalized=top_album.normalized_title,
        image_url=top_album.image_url,
        lastfm_url=top_album.url,
        playcount=top_album.playcount,
    )


def parse_arguments() -> argparse.Namespace:
    def worker_count(value: str) -> int:
        workers = int(value)
        if not 1 <= workers <= MAX_WORKERS:
            raise argparse.ArgumentTypeError(
                f"workers must be between 1 and {MAX_WORKERS}, got {workers}"
            )
        return workers

    parser = argparse.ArgumentParser(
        description="Find missing popular albums using Last.fm data."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache when fetching Last.fm data (still writes cache).",
    )
    parser.add_argument(
        "--limit-artists",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N artists discovered (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        default=DEFAULT_WORKERS,
        help=f"Number of worker threads to use (1-{MAX_WORKERS}, default {DEFAULT_WORKERS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_logging()
    logging.info("Missing Popular Albums script started.")

    api_key = os.environ.get("LASTFM_API_KEY") or CONFIG.get("LASTFM_API_KEY", "")
    if not api_key:
        print(
            "Last.fm API key missing. Set LASTFM_API_KEY environment variable, e.g.:\n"
            'export LASTFM_API_KEY="YOUR_KEY"'
        )
        logging.error("Missing LASTFM_API_KEY. Exiting.")
        sys.exit(1)

    client = LastFMClient(api_key=api_key)
    cache = load_cache(CACHE_FILE) if not args.no_cache else {}
    cache_lock = threading.Lock()
    stats_lock = threading.Lock()
    cache_stats = {"hits": 0, "misses": 0}

    local_artists = scan_library(MUSIC_ROOT)
    if not local_artists:
        logging.warning("No artists discovered in local library.")
        print("No artists discovered in the local library.")
        return

    artist_items = sorted(
        local_artists.values(),
        key=lambda artist: normalize_text(artist.display_name),
    )
    original_count = len(artist_items)
    if args.limit_artists is not None:
        artist_items = artist_items[: args.limit_artists]
        logging.info(
            "Limiting processing to %d artist(s) out of %d due to --limit-artists flag.",
            len(artist_items),
            original_count,
        )

    suggestions: List[AlbumSuggestion] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_artist = {
            executor.submit(
                process_artist,
                artist,
                client,
                cache,
                cache_lock,
                stats_lock,
                cache_stats,
                not args.no_cache,
            ): artist
            for artist in artist_items
        }
        with tqdm(total=len(artist_items), desc="Processing artists", unit="artist") as progress:
            for future in concurrent.futures.as_completed(future_to_artist):
                artist = future_to_artist[future]
                try:
                    suggestion = future.result()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logging.exception("Error processing %s: %s", artist.display_name, exc)
                    suggestion = None
                if suggestion:
                    suggestions.append(suggestion)
                progress.update(1)

    suggestions.sort(key=lambda item: (item.artist_normalized, item.album_normalized))

    try:
        render_html(suggestions, HTML_OUT, total_artists=len(artist_items))
    except Exception:
        print(f"Failed to write HTML output to {HTML_OUT}. See log for details.")
        return

    with cache_lock:
        save_cache(CACHE_FILE, cache)

    summarize_results(len(artist_items), suggestions, cache_stats)


if __name__ == "__main__":
    main()
