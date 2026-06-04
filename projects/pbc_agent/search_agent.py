"""
search_agent.py — wraps Google Custom Search API (primary) with SerpAPI fallback.
Returns a list of {title, link, snippet} dicts per query.
"""

import time
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, SERPAPI_KEY, MAX_RESULTS, SEARCH_DELAY

log = logging.getLogger(__name__)


class SearchResult:
    def __init__(self, title: str, link: str, snippet: str):
        self.title   = title
        self.link    = link
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {"title": self.title, "link": self.link, "snippet": self.snippet}

    def __repr__(self):
        return f"<SearchResult title={self.title!r} link={self.link!r}>"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _google_search(query: str, num: int = MAX_RESULTS) -> list[SearchResult]:
    """Google Custom Search JSON API."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in .env")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx":  GOOGLE_CSE_ID,
        "q":   query,
        "num": min(num, 10),
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        SearchResult(
            title   = i.get("title", ""),
            link    = i.get("link", ""),
            snippet = i.get("snippet", ""),
        )
        for i in items
    ]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _serpapi_search(query: str, num: int = MAX_RESULTS) -> list[SearchResult]:
    """SerpAPI fallback — uses same Google results but different billing."""
    if not SERPAPI_KEY:
        raise ValueError("SERPAPI_KEY must be set in .env for fallback search")

    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_KEY,
        "engine":  "google",
        "q":       query,
        "num":     num,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("organic_results", [])
    return [
        SearchResult(
            title   = r.get("title", ""),
            link    = r.get("link", ""),
            snippet = r.get("snippet", ""),
        )
        for r in results[:num]
    ]


def search(query: str, num: int = MAX_RESULTS) -> list[SearchResult]:
    """
    Try Google Custom Search; fall back to SerpAPI on failure.
    Respects SEARCH_DELAY between calls to avoid rate-limit errors.
    """
    time.sleep(SEARCH_DELAY)
    try:
        results = _google_search(query, num)
        log.info("Google CSE returned %d results for: %s", len(results), query[:80])
        return results
    except Exception as e:
        log.warning("Google CSE failed (%s), trying SerpAPI fallback", e)
        try:
            results = _serpapi_search(query, num)
            log.info("SerpAPI returned %d results for: %s", len(results), query[:80])
            return results
        except Exception as e2:
            log.error("Both search providers failed: %s", e2)
            return []
