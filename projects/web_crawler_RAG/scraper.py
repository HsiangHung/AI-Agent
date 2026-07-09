"""
scraper.py — fetches page content from URLs returned by search results.
Returns clean text for the LLM extractor.
"""

import logging
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Sites that tend to block scrapers or return junk — skip them
BLOCKLIST = [
    "twitter.com", "facebook.com", "instagram.com",
    "reddit.com",  # heavily JS-rendered
    "wsj.com", "nytimes.com",  # hard paywalls
]

MAX_CONTENT_CHARS = 8_000  # clip to keep LLM context manageable


def _is_blocked(url: str) -> bool:
    return any(domain in url for domain in BLOCKLIST)


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove nav/header/footer/ads
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Prefer <article> or <main> content
    main = soup.find("article") or soup.find("main") or soup.find("body")
    text = main.get_text(separator=" ", strip=True) if main else soup.get_text(" ", strip=True)

    # Collapse whitespace
    import re
    text = re.sub(r"\s{2,}", " ", text)
    return text[:MAX_CONTENT_CHARS]


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def fetch_page_text(url: str) -> str:
    """Fetch a URL and return cleaned plain text (up to MAX_CONTENT_CHARS)."""
    if _is_blocked(url):
        log.debug("Skipping blocked URL: %s", url)
        return ""

    try:
        resp = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            log.debug("Non-HTML content at %s (%s), skipping", url, content_type)
            return ""

        return _extract_text(resp.text)
    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return ""


def gather_context(search_results: list, max_pages: int = 3) -> str:
    """
    Fetch top max_pages URLs from search results and concatenate their text.
    Also includes snippets from all results for fallback extraction.
    """
    snippets = "\n".join(
        f"[SNIPPET] {r.title}: {r.snippet}" for r in search_results
    )

    page_texts = []
    for r in search_results[:max_pages]:
        text = fetch_page_text(r.link)
        if text:
            page_texts.append(f"[SOURCE: {r.link}]\n{text}")

    full_context = snippets
    if page_texts:
        full_context += "\n\n" + "\n\n---\n\n".join(page_texts)

    return full_context
