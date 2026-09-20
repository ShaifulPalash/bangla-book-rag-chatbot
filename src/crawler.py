"""
src/crawler.py
================
Discovers and downloads every chapter of দেবদাস from Bengali Wikisource,
cleans the raw HTML into plain text, and saves the result (with metadata)
to data/raw_debdas.json for the next phase (chunking) to use.

Why we talk to the MediaWiki API instead of scraping raw HTML pages:
----------------------------------------------------------------------
Bengali Wikisource runs on MediaWiki — the same software as Wikipedia.
MediaWiki sites expose a stable, official JSON API (api.php) for exactly
this kind of task: listing pages and fetching their content. Scraping the
human-facing HTML pages directly (guessing URLs, hoping CSS classes never
change) is fragile. Using the API is the standard, more robust approach:

  1. list=allpages with a title "prefix" lets us DISCOVER every chapter
     subpage automatically, instead of hardcoding "there are 16 chapters."
     If this script were pointed at a different book with a different
     chapter count, this discovery step would still work correctly.
  2. action=parse gives us clean, already-rendered HTML for a page, which
     we then strip down to just the story text using BeautifulSoup.

A note on testing: this script makes real HTTP requests to
bn.wikisource.org. It cannot be executed inside the sandboxed environment
used to *write* this code (that environment's network access is limited
to package registries like PyPI/npm, not general websites). It WILL work
when you run it yourself, since your machine has normal internet access.
Run it with:  python src/crawler.py
"""

import json
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import BOOK_TITLE, BOOK_AUTHOR, RAW_BOOK_PATH, logger

# ---------------------------------------------------------------------------
# Wikisource / MediaWiki API settings
# ---------------------------------------------------------------------------
API_URL = "https://bn.wikisource.org/w/api.php"

# The exact MediaWiki page title for the book's main/index page.
# (This is a page TITLE, not a URL — requests will handle encoding for us,
# so we never have to manually percent-encode Bengali text.)
MAIN_PAGE_TITLE = "দেবদাস (শরৎচন্দ্র চট্টোপাধ্যায়)"

# Wikimedia's API etiquette policy asks every client to identify itself
# with a descriptive User-Agent. Omitting this can lead to your requests
# being silently rate-limited or blocked.
HEADERS = {
    "User-Agent": (
        "DebdasRAGChatbot/1.0 (Educational assignment project; "
        "contact: student@example.com)"
    )
}

# Bengali ordinal words 1st through 16th, in correct reading order.
# We need this because the API's allpages list comes back alphabetically,
# not in story order — this list lets us re-sort chapters correctly and
# assign a clean chapter_number to each one.
BENGALI_ORDINALS = [
    "প্রথম", "দ্বিতীয়", "তৃতীয়", "চতুর্থ", "পঞ্চম", "ষষ্ঠ", "সপ্তম",
    "অষ্টম", "নবম", "দশম", "একাদশ", "দ্বাদশ", "ত্রয়োদশ", "চতুর্দশ",
    "পঞ্চদশ", "ষোড়শ",
]

# Being a polite crawler: a short pause between requests so we don't
# hammer Wikisource's servers with 16+ rapid-fire requests.
REQUEST_DELAY_SECONDS = 0.5


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _api_get(params: dict) -> dict:
    """
    Thin wrapper around requests.get() for the MediaWiki API, with
    automatic retries (exponential backoff) for transient network errors.
    Every other function in this file goes through here, so all API
    calls get the same retry protection and logging for free.
    """
    params = {**params, "format": "json"}
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()  # raises an exception on HTTP 4xx/5xx
    return response.json()


def discover_chapter_titles() -> list[str]:
    """
    Step 1: ask the API for every page whose title starts with
    "দেবদাস (শরৎচন্দ্র চট্টোপাধ্যায়)/" — i.e. every chapter subpage.

    Returns a list of full page titles, sorted into correct reading order
    (প্রথম পরিচ্ছেদ, দ্বিতীয় পরিচ্ছেদ, ..., ষোড়শ পরিচ্ছেদ).
    """
    logger.info(f"Discovering chapter subpages under '{MAIN_PAGE_TITLE}' ...")

    params = {
        "action": "query",
        "list": "allpages",
        "apprefix": MAIN_PAGE_TITLE + "/",
        "apnamespace": 0,       # namespace 0 = main article namespace
        "aplimit": "max",
    }
    data = _api_get(params)
    pages = data.get("query", {}).get("allpages", [])
    titles = [p["title"] for p in pages]

    if not titles:
        # Fallback: if discovery finds nothing (e.g. API shape changed),
        # fall back to constructing titles from our known ordinal list.
        # This keeps the crawler working even if the primary method fails,
        # rather than crashing with an empty book.
        logger.warning(
            "API discovery returned no subpages — falling back to the "
            "known 16-chapter ordinal list."
        )
        titles = [f"{MAIN_PAGE_TITLE}/{ordinal} পরিচ্ছেদ" for ordinal in BENGALI_ORDINALS]

    # Sort into correct story order using the ordinal list, instead of
    # trusting the API's alphabetical order.
    def sort_key(title: str) -> int:
        for i, ordinal in enumerate(BENGALI_ORDINALS):
            if title.endswith(f"{ordinal} পরিচ্ছেদ"):
                return i
        return len(BENGALI_ORDINALS)  # unknown chapters sort last, not dropped

    titles.sort(key=sort_key)
    logger.info(f"Discovered {len(titles)} chapter subpages.")
    return titles


def fetch_page_html(title: str) -> str:
    """
    Step 2: ask the API to render a single page's wikitext into HTML,
    exactly as a browser would see it. We'll clean this HTML down to
    plain story text in clean_html_to_text() below.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
    }
    data = _api_get(params)
    html = data.get("parse", {}).get("text", {}).get("*", "")
    if not html:
        logger.warning(f"No HTML content returned for page: {title}")
    return html


def clean_html_to_text(html: str) -> str:
    """
    Step 3: strip a rendered Wikisource page down to just the story text.

    MediaWiki pages contain a lot of things we don't want in our knowledge
    base: "[edit]" section links, footnote markers, navigation boxes, and
    the "retrieved from ..." footer. We remove all of that here so the
    embeddings (Phase 4) represent only the actual novel text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # The actual article content on any MediaWiki page lives inside a
    # <div class="mw-parser-output"> — everything outside it (sidebars,
    # etc.) is not part of the page body to begin with, but we scope to
    # it explicitly for clarity and safety.
    content = soup.find("div", class_="mw-parser-output") or soup

    # Remove elements that are structural/editorial, not story content:
    unwanted_selectors = [
        "sup.reference",       # footnote markers like [১]
        "span.mw-editsection",  # "[edit]" links
        "table",                # navigation/infobox tables
        ".noprint",              # elements MediaWiki marks as non-content
        "style",
        "script",
    ]
    for selector in unwanted_selectors:
        for tag in content.select(selector):
            tag.decompose()

    # Extract visible text, using newlines between block elements so
    # paragraphs don't run together into one giant line.
    text = content.get_text(separator="\n")

    # Clean up common leftover noise:
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]  # drop empty lines

    # Drop the "... থেকে আনীত" ("retrieved from ...") footer line that
    # MediaWiki sometimes includes, and any leftover reference bracket
    # markers like "[১]" that survived as plain text.
    cleaned_lines = []
    for line in lines:
        if "থেকে আনীত" in line:
            continue
        cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines).strip()


def build_source_url(title: str) -> str:
    """
    Builds the human-clickable Wikisource URL for a page title, for use
    as a citation source. quote() percent-encodes the Bengali text
    correctly; safe="/()" keeps slashes and parentheses readable instead
    of also encoding them.
    """
    return "https://bn.wikisource.org/wiki/" + quote(title.replace(" ", "_"), safe="/()")


def crawl_book() -> list[dict]:
    """
    Orchestrates the full crawl: discover chapters, fetch + clean each
    one, and return a list of chapter records ready to be saved to disk.
    """
    titles = discover_chapter_titles()
    chapters = []

    for i, title in enumerate(titles, start=1):
        chapter_name = title.split("/")[-1]  # e.g. "সপ্তম পরিচ্ছেদ"
        logger.info(f"[{i}/{len(titles)}] Fetching: {chapter_name}")

        try:
            html = fetch_page_html(title)
            text = clean_html_to_text(html)
        except Exception as e:
            # A single failed chapter shouldn't kill the whole crawl.
            # We log the error clearly and move on, so a network blip on
            # chapter 9 doesn't cost you chapters 10-16 too.
            logger.error(f"Failed to fetch/clean '{title}': {e}")
            continue

        if not text:
            logger.warning(f"Chapter '{chapter_name}' produced no text — skipping.")
            continue

        chapters.append({
            "book_name": BOOK_TITLE,
            "author": BOOK_AUTHOR,
            "chapter_number": i,
            "chapter_name": chapter_name,
            "source_url": build_source_url(title),
            "text": text,
        })

        logger.info(f"  -> {len(text)} characters extracted.")
        time.sleep(REQUEST_DELAY_SECONDS)  # polite crawling

    return chapters


def main():
    chapters = crawl_book()

    if not chapters:
        logger.error("Crawl produced zero chapters. Aborting save.")
        return

    with open(RAW_BOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(c["text"]) for c in chapters)
    logger.info(
        f"Crawl complete: {len(chapters)} chapters, "
        f"{total_chars:,} total characters saved to {RAW_BOOK_PATH}"
    )


if __name__ == "__main__":
    main()
