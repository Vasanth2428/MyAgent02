# Web search tool using Tavily.
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("MultiAgent.WebSearchTool")


def get_tavily_api_key() -> Optional[str]:
    """Get Tavily API key from environment."""
    return os.getenv("TAVILY_API_KEY")


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Perform web search using Tavily API."""
    api_key = get_tavily_api_key()
    if not api_key:
        logger.warning("TAVILY_API_KEY not set, using mock search")
        return mock_web_search(query)
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        return response.get("results", [])
    except ImportError:
        logger.warning("tavily-python not installed, using mock search")
        return mock_web_search(query)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []


def mock_web_search(query: str) -> List[Dict]:
    """Fallback web search using DuckDuckGo when Tavily is unavailable."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return [{"title": "No Results", "url": "#", "content": f"No results found for '{query}'."}]
            return [
                {
                    "title": r.get("title", "No Title"),
                    "url": r.get("href", "#"),
                    "content": r.get("body", "")
                }
                for r in results
            ]
    except Exception as e:
        logger.error(f"DDGS search error: {e}")
        return [{"title": "Search Unavailable", "url": "#", "content": f"Web search failed: {e}"}]


def format_search_results(results: List[Dict]) -> str:
    """Format search results as context string."""
    if not results:
        return "No web search results found."
    formatted = "\n\n".join([f"- {r.get('title', 'Unknown')}: {r.get('content', '')}" for r in results])
    return formatted
