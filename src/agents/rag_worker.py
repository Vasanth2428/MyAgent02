# RAG worker node - strictly document-only answers.
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.core.config import RAG_WORKER_MODEL_PRIMARY, RAG_WORKER_MODEL_FALLBACK
from src.core.model_provider import build_model_with_fallback, message_text

logger = logging.getLogger("MultiAgent.RAGWorker")

RAG_SYSTEM_PROMPT = """You are a RAG specialist. You can ONLY use the document_search tool to answer questions.
STRICT: Answer ONLY from the documents returned by the retriever. Do not use parametric knowledge.
If the answer is not found in the provided documents, respond with:
"I don't know based on the provided documents."

Never invent information. Use only what you find in the document search results.
"""


def get_reasoning_model():
    """Get the configured LLM model for document reasoning."""
    return build_model_with_fallback(
        "rag_worker",
        RAG_WORKER_MODEL_PRIMARY,
        RAG_WORKER_MODEL_FALLBACK,
        temperature=0,
        api_key_envs=("GROQ_CORE_KEY", "AGENT_API_KEY"),
    )


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
        from src.core.retriever import WeaviateRetriever
        retriever = WeaviateRetriever()
        tool = DocumentRetrieverTool(retriever)
        document_tool = tool.search
    
    current_task = state.get("current_task", "")
    scratchpad = state.get("scratchpad", "")
    worker_complete = state.get("worker_complete", {})
    worker_outputs = state.get("worker_outputs", {})
    
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
            "messages": [AIMessage(content="No query provided.", name="rag_worker")],
            "scratchpad": scratchpad,
            "worker_complete": {"rag_worker": True},
            "worker_outputs": {"rag_worker": "No query provided."},
            "worker_type": "rag_worker",
            "next_agent": "supervisor"
        }
    
    try:
        print(f"\n[RAG WORKER] Querying database for: '{target_query}'")
        results = document_tool(target_query)
        results = truncate_results(results)
        
        if not results:
            print("[RAG WORKER] No documents found in database.")
            no_doc_msg = "I don't know based on the provided documents."
            updated_scratchpad = scratchpad + f"\n- [RAG Worker]: {no_doc_msg}"
            return {
                "messages": [AIMessage(content=no_doc_msg, name="rag_worker")],
                "scratchpad": updated_scratchpad,
                "worker_complete": {"rag_worker": True},
                "worker_outputs": {"rag_worker": no_doc_msg},
                "worker_type": "rag_worker",
                "next_agent": "supervisor"
            }
        
        print(f"[RAG WORKER] Retrieved {len(results)} document segments. Generating final answer...")
        context = "\n\n".join([f"Document {i+1}:\n{validate_tool_output(r.get('text', ''))}" for i, r in enumerate(results)])
        
        model = get_reasoning_model()
        response = model.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {target_query}")
        ])
        
        safe_response = validate_tool_output(message_text(response))
        print(f"[RAG WORKER] Response:\n{safe_response}")
        
        updated_scratchpad = scratchpad + f"\n- [RAG Worker]: {safe_response}"
        return {
            "messages": [AIMessage(content=safe_response, name="rag_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"rag_worker": True},
            "worker_outputs": {"rag_worker": safe_response},
            "worker_type": "rag_worker",
            "next_agent": "supervisor"
        }
    except Exception as e:
        logger.error(f"RAG worker error: {e}")
        err_msg = "Error searching documents. Please try again."
        updated_scratchpad = scratchpad + f"\n- [RAG Worker]: {err_msg}"
        return {
            "messages": [AIMessage(content=err_msg, name="rag_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"rag_worker": True},
            "worker_outputs": {"rag_worker": err_msg},
            "worker_type": "rag_worker",
            "next_agent": "supervisor"
        }
