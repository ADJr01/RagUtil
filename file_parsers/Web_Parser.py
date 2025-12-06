# file: web_scraper.py

import time
import random
import logging
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/124.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) Trident/7.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/122 Mobile",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/125 Safari/537.36"
]

DEFAULT_PROXIES = [
    "http://159.65.245.255:80",
    "http://34.90.106.238:3128",
    "http://178.128.163.157:8080",
    "http://51.158.68.68:8811",
    "http://20.210.113.32:80"
]


def get_random_user_agent() -> str:
    """Returns random user-agent."""
    return random.choice(DEFAULT_USER_AGENTS)


def get_proxy() -> dict | None:
    """Returns working proxy dict or None."""
    for proxy in DEFAULT_PROXIES:
        try:
            r = requests.get(
                "http://www.google.com",
                proxies={"http": proxy, "https": proxy},
                timeout=2
            )
            if r.status_code == 200:
                return {"http": proxy, "https": proxy}
        except Exception:
            continue
    return None


def request_with_rotations(url: str) -> requests.Response:
    """Handles UA rotation, proxy rotation, retry on block."""
    def attempt_request():
        headers = {"User-Agent": get_random_user_agent()}
        proxy = get_proxy()
        return requests.get(url, headers=headers, proxies=proxy, timeout=10)

    try:
        resp = attempt_request()
        if resp.status_code in (403, 429):
            time.sleep(2)
            resp = attempt_request()
        return resp
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")


def search_google(query: str) -> list[str]:
    encoded = quote(query)
    url = f"https://www.google.com/search?q={encoded}"
    resp = request_with_rotations(url)

    if resp.status_code not in (200, 302):
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.startswith("/url?q="):
            real_url = href.split("/url?q=")[1].split("&")[0]
            results.append(real_url)
            if len(results) >= 15:
                break

    return results


def search_duckduckgo(query: str) -> list[str]:
    encoded = quote(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    resp = request_with_rotations(url)

    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for a in soup.find_all("a", class_="result__a"):
        href = a.get("href")
        if href:
            results.append(href)
        if len(results) >= 15:
            break

    return results


def perform_search(query: str, search_engine: str = "google") -> list[str]:
    engine = search_engine.lower()

    if engine == "google":
        try:
            out = search_google(query)
            if out:
                return out
        except Exception:
            pass
        return search_duckduckgo(query)

    if engine == "duckduckgo":
        try:
            out = search_duckduckgo(query)
            if out:
                return out
        except Exception:
            pass
        return search_google(query)

    raise ValueError("Unsupported search engine")


def fetch_website_content(url: str) -> str | None:
    time.sleep(random.uniform(1, 3))

    try:
        resp = request_with_rotations(url)
        if resp.status_code != 200:
            return None

        text = resp.text.lower()
        blocked_signals = ["captcha", "cloudflare", "access denied", "robot verification"]
        if any(b in text for b in blocked_signals):
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        raw_text = soup.get_text(" ", strip=True)
        return " ".join(raw_text.split())

    except Exception:
        return None


def main(query: str, search_engine: str = "google", limit: int = 10) -> list[str]:
    results = []
    urls = perform_search(query, search_engine)

    for url in urls[:limit]:
        content = fetch_website_content(url)
        if content:
            results.append(content)

    return results



