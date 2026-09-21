"""
src/crawler.py
==============

Downloads all chapters of দেবদাস from Bengali Wikisource through the
MediaWiki API, cleans the rendered HTML, validates chapter completeness,
and saves the result to data/raw_debdas.json.
"""

from __future__ import annotations

import json
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    BOOK_AUTHOR,
    BOOK_TITLE,
    EXPECTED_CHAPTER_COUNT,
    RAW_BOOK_PATH,
    logger,
)


API_URL = "https://bn.wikisource.org/w/api.php"

MAIN_PAGE_TITLE = "দেবদাস (শরৎচন্দ্র চট্টোপাধ্যায়)"

HEADERS = {
    "User-Agent": (
        "DebdasRAGChatbot/1.1 "
        "(educational Bengali RAG project)"
    )
}


BENGALI_ORDINALS = [
    "প্রথম",
    "দ্বিতীয়",
    "তৃতীয়",
    "চতুর্থ",
    "পঞ্চম",
    "ষষ্ঠ",
    "সপ্তম",
    "অষ্টম",
    "নবম",
    "দশম",
    "একাদশ",
    "দ্বাদশ",
    "ত্রয়োদশ",
    "চতুর্দশ",
    "পঞ্চদশ",
    "ষোড়শ",
]

REQUEST_DELAY_SECONDS = 0.5


def chapter_number_from_title(title: str) -> int | None:
    """Return the Bengali ordinal chapter number encoded in a page title."""

    for index, ordinal in enumerate(BENGALI_ORDINALS, start=1):
        if title.endswith(f"{ordinal} পরিচ্ছেদ"):
            return index

    return None


@retry(
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=20,
    ),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _api_get(params: dict) -> dict:
    """Perform one MediaWiki API request with retry protection."""

    request_params = {
        **params,
        "format": "json",
    }

    response = requests.get(
        API_URL,
        params=request_params,
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            f"MediaWiki API error: {data['error']}"
        )

    return data


def discover_chapter_titles() -> list[str]:
    """
    Discover all chapter subpages using MediaWiki pagination.

    Only pages matching the expected Bengali chapter naming convention
    are accepted.
    """

    logger.info(
        "Discovering chapter subpages under %r...",
        MAIN_PAGE_TITLE,
    )

    titles: list[str] = []
    continue_token: dict | None = None

    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apprefix": MAIN_PAGE_TITLE + "/",
            "apnamespace": 0,
            "aplimit": "max",
        }

        if continue_token:
            params.update(continue_token)

        data = _api_get(params)

        pages = data.get("query", {}).get("allpages", [])

        for page in pages:
            title = page["title"]

            if chapter_number_from_title(title) is not None:
                titles.append(title)

        continue_token = data.get("continue")

        if not continue_token:
            break

    # Remove duplicates while preserving discovery order.
    titles = list(dict.fromkeys(titles))

    if not titles:
        logger.warning(
            "API discovery returned no chapter pages. "
            "Using the expected 16 chapter titles as a fallback."
        )

        titles = [
            f"{MAIN_PAGE_TITLE}/{ordinal} পরিচ্ছেদ"
            for ordinal in BENGALI_ORDINALS
        ]

    titles.sort(
        key=lambda title: chapter_number_from_title(title) or 999
    )

    if len(titles) != EXPECTED_CHAPTER_COUNT:
        raise RuntimeError(
            "Chapter discovery is incomplete. "
            f"Expected {EXPECTED_CHAPTER_COUNT} chapters but found "
            f"{len(titles)}."
        )

    logger.info(
        "Successfully discovered %d chapters.",
        len(titles),
    )

    return titles


def fetch_page_html(title: str) -> str:
    """Fetch rendered HTML for one Wikisource page."""

    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
    }

    data = _api_get(params)

    html = (
        data.get("parse", {})
        .get("text", {})
        .get("*", "")
    )

    if not html:
        raise RuntimeError(
            f"MediaWiki returned no HTML for page: {title}"
        )

    return html


def clean_html_to_text(html: str) -> str:
    """Convert rendered Wikisource HTML into clean story text."""

    soup = BeautifulSoup(html, "html.parser")

    content = soup.find(
        "div",
        class_="mw-parser-output",
    ) or soup

    unwanted_selectors = [
        "sup.reference",
        "span.mw-editsection",
        "table",
        ".noprint",
        ".reference",
        ".reflist",
        "ol.references",
        "style",
        "script",
    ]

    for selector in unwanted_selectors:
        for tag in content.select(selector):
            tag.decompose()

    text = content.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    cleaned_lines = []

    for line in lines:
        # Remove MediaWiki footer noise.
        if "থেকে আনীত" in line:
            continue

        cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines).strip()


def build_source_url(title: str) -> str:
    """Build the human-readable Wikisource URL for a chapter."""

    encoded_title = quote(
        title.replace(" ", "_"),
        safe="/()",
    )

    return (
        "https://bn.wikisource.org/wiki/"
        + encoded_title
    )


def crawl_book() -> list[dict]:
    """
    Crawl every expected chapter.

    A partial crawl is treated as an error rather than silently producing
    an incomplete knowledge base.
    """

    titles = discover_chapter_titles()

    chapters: list[dict] = []
    failed_chapters: list[str] = []

    for position, title in enumerate(titles, start=1):
        chapter_number = chapter_number_from_title(title)
        chapter_name = title.split("/")[-1]

        logger.info(
            "[%d/%d] Fetching %s...",
            position,
            len(titles),
            chapter_name,
        )

        try:
            html = fetch_page_html(title)
            text = clean_html_to_text(html)

            if not text:
                raise RuntimeError(
                    "Chapter produced empty cleaned text."
                )

            if chapter_number is None:
                raise RuntimeError(
                    "Could not determine chapter number."
                )

            chapters.append(
                {
                    "book_name": BOOK_TITLE,
                    "author": BOOK_AUTHOR,
                    "chapter_number": chapter_number,
                    "chapter_name": chapter_name,
                    "source_url": build_source_url(title),
                    "text": text,
                }
            )

            logger.info(
                "  -> %d characters extracted.",
                len(text),
            )

        except Exception as exc:
            logger.error(
                "Failed to fetch %s: %s",
                chapter_name,
                exc,
            )
            failed_chapters.append(chapter_name)

        time.sleep(REQUEST_DELAY_SECONDS)

    if failed_chapters:
        raise RuntimeError(
            "Crawl incomplete. Failed chapters: "
            + ", ".join(failed_chapters)
            + ". No incomplete raw book file was written."
        )

    chapters.sort(
        key=lambda chapter: chapter["chapter_number"]
    )

    if len(chapters) != EXPECTED_CHAPTER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CHAPTER_COUNT} chapters but collected "
            f"{len(chapters)}."
        )

    return chapters


def main() -> None:
    """Run the complete crawler."""

    chapters = crawl_book()

    with open(
        RAW_BOOK_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chapters,
            file,
            ensure_ascii=False,
            indent=2,
        )

    total_chars = sum(
        len(chapter["text"])
        for chapter in chapters
    )

    logger.info(
        "Crawl complete: %d chapters, %d total characters saved to %s.",
        len(chapters),
        total_chars,
        RAW_BOOK_PATH,
    )


if __name__ == "__main__":
    main()