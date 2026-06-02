import sys
import os
import asyncio
import logging

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("weaviate").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from core.retriever import WeaviateRetriever
from core.engine import RAGContextEngine
from src.main import run_query

async def main():
    print("Initializing components...")
    retriever = WeaviateRetriever()
    engine = RAGContextEngine(retriever)
    
    question = "What is 15 + 25 * 3?"
    
    # 1. Simple RAG Mode
    print("\n" + "="*60)
    print("1. SIMULATING PIPELINE: SIMPLE RAG (mode='normal')")
    print("="*60)
    try:
        res_normal = await engine.ask_async(question, mode="normal")
        print("\n--- Simple RAG Result ---")
        print("Answer:", res_normal.get("response"))
    except Exception as e:
        print("Error in Simple RAG simulation:", e)
        
    # 2. Context Engine Mode
    print("\n" + "="*60)
    print("2. SIMULATING PIPELINE: CONTEXT ENGINE (mode='context_engine')")
    print("="*60)
    try:
        res_ce = await engine.ask_async(question, mode="context_engine")
        print("\n--- Context Engine Result ---")
        print("Answer:", res_ce.get("response"))
    except Exception as e:
        print("Error in Context Engine simulation:", e)
        
    # 3. Multi-Agent System (New Agentic Loop)
    print("\n" + "="*60)
    print("3. SIMULATING PIPELINE: MULTI-AGENT SYSTEM")
    print("="*60)
    try:
        res_ma = run_query(question)
        print("\n--- Multi-Agent Result ---")
        answer = res_ma.get('final_answer', '') or (res_ma.get('messages', [{}])[-1].content if res_ma.get('messages') else 'No answer')
        print("Answer:", answer)
    except Exception as e:
        print("Error in Multi-Agent simulation:", e)
        
    # Close retriever
    engine.close()

if __name__ == "__main__":
    asyncio.run(main())
