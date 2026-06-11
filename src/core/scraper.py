"""
Web Scraper - Fetching Content from URLs

This module safely fetches and extracts text from web pages. It includes security
measures to prevent SSRF (Server-Side Request Forgery) attacks that try to
access internal network resources.

Features:
- Extracts readable text from HTML
- Validates URLs to block access to private/internal networks
- Supports both sync and async fetching
- Handles multiple concurrent requests safely
"""

import re
import logging
import asyncio
import aiohttp
import requests
from html.parser import HTMLParser
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import ipaddress

logger = logging.getLogger("RAG.Scraper")

_aiohttp_session: Optional[aiohttp.ClientSession] = None
_executor = ThreadPoolExecutor(max_workers=10)


def _is_private_ip(ip_str: str) -> bool:
    """Check if IP address is private/internal and should be blocked for security."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


def _validate_url_for_ssrf(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validates URL for SSRF protection.
    Returns (is_valid, error_message, pinned_ip).
    Blocks private IPs, localhost, and internal network ranges.
    The pinned_ip is returned so callers can pin the connection to it,
    preventing DNS rebinding attacks (SEC-02: TOCTOU fix).
    """
    from urllib.parse import urlparse
    import socket

    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL", None

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        return False, "No hostname in URL", None

    blocked_hostnames = {'localhost', '127.0.0.1', '0.0.0.0', '::1', 'localhost.localdomain'}
    if hostname.lower() in blocked_hostnames:
        return False, f"Blocked internal hostname: {hostname}", None

    first_public_ip = None
    try:
        addrinfos = socket.getaddrinfo(hostname, parsed.port or 80)
        for family, _, _, _, sockaddr in addrinfos:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                return False, f"Blocked private IP: {ip_str}", None
            if first_public_ip is None:
                first_public_ip = ip_str
    except socket.gaierror:
        return False, f"Could not resolve hostname: {hostname}", None
    except Exception as e:
        return False, f"DNS validation error: {e}", None

    return True, "", first_public_ip

async def _get_aiohttp_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=10.0)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    return _aiohttp_session

async def close_aiohttp_session():
    global _aiohttp_session
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()

class HTMLTextExtractor(HTMLParser):
    """
    Custom HTMLParser that extracts readable body text, filtering out style/script tags
    and preserving structural spacing for block elements.
    """
    def __init__(self):
        super().__init__()
        self.record = True
        self.text_parts = []
        self.ignored_tags = {"script", "style", "head", "title", "meta", "link", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.ignored_tags:
            self.record = False
        # Inject linebreaks for block-level elements
        if tag.lower() in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.ignored_tags:
            self.record = True

    def handle_data(self, data):
        if self.record:
            text = data.strip()
            if text:
                # Standardize whitespace
                text = re.sub(r'\s+', ' ', text)
                self.text_parts.append(text)

    def get_text(self) -> str:
        content = " ".join(self.text_parts)
        # Clean up excessive newlines
        content = re.sub(r'\n\s*\n', '\n', content)
        return content.strip()


def scrape_web_page(url: str, max_chars: int = 6000) -> str:
    """
    Fetches the HTML content of the URL, extracts clean body text, and truncates appropriately.
    Synchronous wrapper around async implementation using run_in_executor.
    """
    url = url.strip()
    if not url:
        return "Error: Scrape request received an empty URL."

    # SSRF protection (SEC-02: returns pinned IP to prevent DNS rebinding)
    is_valid, error_msg, pinned_ip = _validate_url_for_ssrf(url)
    if not is_valid:
        logger.warning(f"SSRF blocked: {error_msg}")
        return f"Error: {error_msg}"

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        logger.info(f"Crawling web page (sync): {url}")
        from urllib.parse import urlparse, urlunparse
        if pinned_ip:
            parsed_url = urlparse(url)
            pinned_url = urlunparse(parsed_url._replace(netloc=f"{pinned_ip}:{parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)}"))
            headers["Host"] = parsed_url.hostname
            response = requests.get(pinned_url, headers=headers, timeout=10.0, verify=parsed_url.scheme == 'https')
        else:
            response = requests.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            if "text/plain" in content_type:
                return response.text[:max_chars]
            return f"Error: Unsupported content-type '{content_type}'. Only HTML pages can be scraped."

        parser = HTMLTextExtractor()
        parser.feed(response.text)
        text = parser.get_text()

        if not text:
            return "Warning: Page fetched successfully but no text content was found in the body."

        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n... [Truncated. Total length: {len(text)} characters] ..."

        return text
    except requests.exceptions.Timeout:
        logger.error(f"Timeout scraping URL: {url}")
        return f"Error: The request to fetch '{url}' timed out."
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error scraping URL: {url} - {http_err}")
        try:
            status = response.status_code
        except Exception:
            status = "unknown"
        return f"Error: HTTP request failed with status: {status}."
    except Exception as e:
        logger.error(f"Unexpected error scraping URL: {url} - {e}")
        return f"Error: Failed to fetch or parse page: {type(e).__name__}: {e}"


async def scrape_multiple_pages_async(urls: List[str], max_chars: int = 6000, max_concurrent: int = 5) -> List[str]:
    """
    Asynchronously fetches multiple URLs concurrently with controlled concurrency.
    Returns list of scraped texts in the same order as input URLs.
    """
    import aiohttp

    async def scrape_with_semaphore(url: str, sem: asyncio.Semaphore) -> str:
        async with sem:
            return await scrape_web_page_async(url, max_chars)

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [scrape_with_semaphore(url, semaphore) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r if isinstance(r, str) else f"Error: {r}" for r in results]


async def scrape_web_page_async(url: str, max_chars: int = 6000) -> str:
    """
    Asynchronously fetches the HTML content of the URL, extracts clean body text, and truncates appropriately.
    """
    url = url.strip()
    if not url:
        return "Error: Scrape request received an empty URL."

    # SSRF protection
    is_valid, error_msg, pinned_ip = _validate_url_for_ssrf(url)
    if not is_valid:
        logger.warning(f"SSRF blocked: {error_msg}")
        return f"Error: {error_msg}"

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = None
    try:
        logger.info(f"Crawling web page (async): {url}")
        session = await _get_aiohttp_session()
        from urllib.parse import urlparse, urlunparse
        if pinned_ip:
            parsed_url = urlparse(url)
            pinned_url = urlunparse(parsed_url._replace(netloc=f"{pinned_ip}:{parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)}"))
            headers["Host"] = parsed_url.hostname
            # Use ssl=False to ignore hostname mismatch when connecting directly to IP
            async with session.get(pinned_url, headers=headers, ssl=False) as resp:
                response = resp
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    if "text/plain" in content_type:
                        text = await response.text()
                        return text[:max_chars]
                    return f"Error: Unsupported content-type '{content_type}'. Only HTML pages can be scraped."
                html = await response.text()
        else:
            async with session.get(url, headers=headers) as resp:
                response = resp
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    if "text/plain" in content_type:
                        text = await response.text()
                        return text[:max_chars]
                    return f"Error: Unsupported content-type '{content_type}'. Only HTML pages can be scraped."
                html = await response.text()

        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()

        if not text:
            return "Warning: Page fetched successfully but no text content was found in the body."

        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n... [Truncated. Total length: {len(text)} characters] ..."

        return text
    except asyncio.TimeoutError:
        logger.error(f"Timeout scraping URL: {url}")
        return f"Error: The request to fetch '{url}' timed out."
    except aiohttp.ClientResponseError as http_err:
        logger.error(f"HTTP error scraping URL: {url} - {http_err}")
        status = getattr(response, "status", "unknown") if response else "unknown"
        return f"Error: HTTP request failed with status: {status}."
    except Exception as e:
        logger.error(f"Unexpected error scraping URL: {url} - {e}")
        return f"Error: Failed to fetch or parse page: {type(e).__name__}: {e}"

