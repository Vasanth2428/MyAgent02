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
    """Mock web search when Tavily is unavailable."""
    query_lower = query.lower()
    if "google" in query_lower:
        return [{"title": "Google Cloud AI", "url": "https://cloud.google.com", "content": "Google Cloud announces new AI infrastructure accelerators."}]
    elif "nvidia" in query_lower or "nvda" in query_lower:
        return [{"title": "NVIDIA GTC 2026", "url": "https://nvidia.com", "content": "Nvidia releases next-gen Blackwell Ultra GPU architectures."}]
    elif "apple" in query_lower or "iphone" in query_lower:
        return [{"title": "Apple WWDC 2026", "url": "https://apple.com", "content": "Apple announces iOS 20 with deep neural agentic capabilities."}]
    else:
        return [{"title": "Search Result", "url": "#", "content": f"No specific results for '{query}'. This is a mock response."}]


def format_search_results(results: List[Dict]) -> str:
    """Format search results as context string."""
    if not results:
        return "No web search results found."
    formatted = "\n\n".join([f"- {r.get('title', 'Unknown')}: {r.get('content', '')}" for r in results])
    return formatted
