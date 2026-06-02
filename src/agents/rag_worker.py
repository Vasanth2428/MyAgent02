# RAG worker node - strictly document-only answers.
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.RAGWorker")

RAG_SYSTEM_PROMPT = """You are a RAG specialist. You can ONLY use the document_search tool to answer questions.
STRICT: Answer ONLY from the documents returned by the retriever. Do not use parametric knowledge.
If the answer is not found in the provided documents, respond with:
"I don't know based on the provided documents."

Never invent information. Use only what you find in the document search results.
"""


def get_reasoning_model():
    """Get the LLM model for complex reasoning via Groq."""
    model_name = os.getenv("REASONING_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("AGENT_API_KEY")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)


def rag_worker_node(state: dict, document_tool: callable = None) -> dict:
    """
    RAG worker that searches documents and answers based solely on document content.
    
    Args:
        state: Current state with messages and context_notes.
        document_tool: Function to search documents (callable).
    """
    from src.tools.safety_filters import sanitize_user_input, validate_tool_output, truncate_results
    
    if document_tool is None:
        from src.tools.document_tool import DocumentRetrieverTool
        from core.retriever import WeaviateRetriever
        retriever = WeaviateRetriever()
        tool = DocumentRetrieverTool(retriever)
        document_tool = tool.search
    
    messages = state.get("messages", [])
    
    last_user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_query = sanitize_user_input(msg.get("content", ""))
            break
        elif isinstance(msg, HumanMessage):
            last_user_query = sanitize_user_input(msg.content)
            break
    
    if not last_user_query:
        return {"final_answer": "No query provided.", "next_agent": "FINISH"}
    
    try:
        print(f"\n[RAG WORKER] Querying database for: '{last_user_query}'")
        results = document_tool(last_user_query)
        results = truncate_results(results)
        
        if not results:
            print("[RAG WORKER] No documents found in database.")
            return {
                "messages": [AIMessage(content="I don't know based on the provided documents.")],
                "next_agent": "FINISH"
            }
        
        print(f"[RAG WORKER] Retrieved {len(results)} document segments. Generating final answer...")
        context = "\n\n".join([f"Document {i+1}:\n{validate_tool_output(r.get('text', ''))}" for i, r in enumerate(results)])
        
        model = get_reasoning_model()
        response = model.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {last_user_query}")
        ])
        
        safe_response = validate_tool_output(response.content)
        print(f"[RAG WORKER] Response:\n{safe_response}")
        
        return {
            "messages": [AIMessage(content=safe_response)],
            "next_agent": "FINISH"
        }
    except Exception as e:
        logger.error(f"RAG worker error: {e}")
        return {
            "messages": [AIMessage(content="Error searching documents. Please try again.")],
            "next_agent": "FINISH"
        }
