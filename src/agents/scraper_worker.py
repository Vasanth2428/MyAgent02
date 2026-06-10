# Scraper worker node - crawls URLs and extracts clean text content.
import os
import re
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.ScraperWorker")

SCRAPER_SYSTEM_PROMPT = """You are a web scraper analysis specialist.
Your job is to read the clean text scraped from a webpage and summarize the key facts that answer the user's inquiry.
Keep your output structured, clean, and focus only on the facts related to the query.
"""


from src.core.config import SCRAPER_WORKER_MODEL_PRIMARY, SCRAPER_WORKER_MODEL_FALLBACK

def get_reasoning_model():
    """Get the LLM model for complex reasoning via Groq."""
    api_key = os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=SCRAPER_WORKER_MODEL_PRIMARY, temperature=0, api_key=api_key)
    fallback = ChatGroq(model=SCRAPER_WORKER_MODEL_FALLBACK, temperature=0, api_key=api_key)
    return primary.with_fallbacks([fallback])


def safe_truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncates text up to max_chars without cutting mid-word or mid-sentence."""
    if len(text) <= max_chars:
        return text
    
    # Try to find a sentence/paragraph boundary near the end
    # Look for last period, question mark, exclamation mark followed by space or newline
    # within the last 500 characters of the limit.
    slice_area = text[:max_chars]
    boundaries = [m.start() for m in re.finditer(r'[.!?](\s+|\n|$)', slice_area)]
    if boundaries:
        # Get the latest boundary that is not too far back (e.g., within 500 chars of max_chars)
        latest = boundaries[-1]
        if max_chars - latest <= 500:
            return text[:latest + 1]
            
    # Fallback to last whitespace/word boundary
    last_space = slice_area.rfind(' ')
    if last_space != -1 and max_chars - last_space <= 100:
        return text[:last_space]
        
    return slice_area


async def scraper_worker_node(state: dict, scraper_tool: callable = None) -> dict:
    """
    Scraper worker that fetches text content from target URLs.
    """
    from src.core.scraper import scrape_web_page_async
    from src.tools.safety_filters import sanitize_user_input, validate_tool_output
    
    current_task = state.get("current_task", "")
    scratchpad = state.get("scratchpad", "")
    
    target_query = current_task if current_task else ""
    if not target_query:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                target_query = sanitize_user_input(msg.get("content", ""))
                break
            elif isinstance(msg, HumanMessage):
                target_query = sanitize_user_input(msg.content)
                break
    
    if not target_query:
        return {
            "messages": [AIMessage(content="No query or URL provided.", name="scraper_worker")],
            "scratchpad": scratchpad,
            "worker_complete": {"scraper_worker": True},
            "worker_outputs": {"scraper_worker": "No query or URL provided."},
            "worker_type": "scraper_worker",
            "next_agent": "supervisor"
        }
    
    url_match = re.search(r'https?://[^\s/$.?#].[^\s]*', target_query, re.IGNORECASE)
    url = url_match.group(0) if url_match else ""
    
    if not url:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            found_url = re.search(r'https?://[^\s/$.?#].[^\s]*', content, re.IGNORECASE)
            if found_url:
                url = found_url.group(0)
                break
                
    if not url:
        err_msg = "Error: Scraper worker could not locate a valid URL to fetch."
        print(f"[SCRAPER WORKER] {err_msg}")
        updated_scratchpad = scratchpad + f"\n- [Scraper Worker]: {err_msg}"
        return {
            "messages": [AIMessage(content=err_msg, name="scraper_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"scraper_worker": True},
            "worker_outputs": {"scraper_worker": err_msg},
            "worker_type": "scraper_worker",
            "next_agent": "supervisor"
        }
        
    try:
        print(f"\n[SCRAPER WORKER] Scraped content requested for: '{url}'")
        
        if scraper_tool is None:
            raw_content = await scrape_web_page_async(url)
        else:
            raw_content = scraper_tool(url)
            
        if raw_content.startswith("Error:"):
            print(f"[SCRAPER WORKER] Scraper tool failed: {raw_content}")
            updated_scratchpad = scratchpad + f"\n- [Scraper Worker]: Scrape failed for {url}: {raw_content}"
            return {
                "messages": [AIMessage(content=raw_content, name="scraper_worker")],
                "scratchpad": updated_scratchpad,
                "worker_complete": {"scraper_worker": True},
                "worker_outputs": {"scraper_worker": raw_content},
                "worker_type": "scraper_worker",
                "next_agent": "supervisor"
            }
            
        print(f"[SCRAPER WORKER] Successfully fetched {len(raw_content)} chars. Compacting and summarizing...")
        
        model = get_reasoning_model()
        
        prompt_content = safe_truncate_text(raw_content, 4000)
        summary_prompt = (
            f"Please read the following scraped web page content from {url} and extract "
            f"all information relevant to the user inquiry: '{target_query}'\n\n"
            f"Scraped content:\n{prompt_content}"
        )
        
        response = model.invoke([
            SystemMessage(content=SCRAPER_SYSTEM_PROMPT),
            HumanMessage(content=summary_prompt)
        ])
        
        safe_response = validate_tool_output(response.content)
        print(f"[SCRAPER WORKER] Scrape analysis complete.")
        
        updated_scratchpad = scratchpad + f"\n- [Scraper Worker]: Content from {url}:\n{safe_response}"
        return {
            "messages": [AIMessage(content=f"Scraped and analyzed {url}", name="scraper_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"scraper_worker": True},
            "worker_outputs": {"scraper_worker": safe_response},
            "worker_type": "scraper_worker",
            "next_agent": "supervisor"
        }
    except Exception as e:
        logger.error(f"Scraper worker error: {e}")
        err_msg = f"Error scraping web page {url}. Please try again."
        updated_scratchpad = scratchpad + f"\n- [Scraper Worker]: {err_msg}"
        return {
            "messages": [AIMessage(content=err_msg, name="scraper_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"scraper_worker": True},
            "worker_outputs": {"scraper_worker": err_msg},
            "worker_type": "scraper_worker",
            "next_agent": "supervisor"
        }