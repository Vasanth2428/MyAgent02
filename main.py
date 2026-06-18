"""
RAG Context Engine API

This is the main entry point for the RAG system. It starts a web server with endpoints for:
- Uploading documents (PDF and text files)
- Asking questions (with both regular and streaming responses)
- Checking system statistics
- Viewing conversation history
- Loading the web interface
"""

import os
import io
import json
import logging
import traceback
import psutil
import asyncio
import sys
import sentence_transformers
import transformers

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

print(f"DEBUG: sys.executable = {sys.executable}", flush=True)
print(f"DEBUG: sentence-transformers = {sentence_transformers.__version__}", flush=True)
print(f"DEBUG: transformers = {transformers.__version__}", flush=True)
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pypdf import PdfReader
from dotenv import load_dotenv

def _serialize_event(event):
    """
    Serialize an event dict to JSON, handling non-serializable LangChain message objects.
    AIMessage, HumanMessage, ToolMessage, etc. are converted to their content string.
    """
    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        # Handle LangChain BaseMessage subclasses (AIMessage, HumanMessage, etc.)
        if hasattr(value, "content"):
            return str(value.content)
        # Leave primitives as-is
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return str(value)
    
    try:
        return json.dumps(_convert(event))
    except Exception:
        # Fallback: try to convert the entire event
        return json.dumps(_convert(event))

# Load environment variables from config directory
load_dotenv(dotenv_path="config/.env", override=True)

# Load environment variables before any local module imports
load_dotenv(override=True)

import uuid

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP, PipelineConfig
from src.core.retriever import WeaviateRetriever
from src.core.engine import RAGContextEngine
from src.core.splitter import RecursiveCharacterSplitter
from src.core.scraper import close_aiohttp_session

# Configure Application Logging
import logging.handlers
from src.core.logging_setup import SessionFileHandler, session_id_var
os.makedirs("logs", exist_ok=True)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

file_handler = SessionFileHandler(
    default_log_path="logs/rag_engine.log",
    log_dir="logs/sessions",
    formatter=formatter
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("RAG-API")

# Silence noisy library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("weaviate").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Global instances initialized during lifespan
rag = None
retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes the heavy ML models and database connections once.
    """
    global rag, retriever
    try:
        logger.info("Starting Modular RAG Context Engine...")
        retriever = WeaviateRetriever()
        pipeline_config = PipelineConfig.from_env()
        logger.info(f"Pipeline config: hyde={pipeline_config.enable_hyde}, expansion={pipeline_config.enable_expansion}, reranking={pipeline_config.enable_reranking}, compression={pipeline_config.enable_compression}")
        
        from src.graph.checkpointer import setup_async_checkpointer
        
        async with setup_async_checkpointer() as checkpointer:
            logger.info("Pre-warming local embedding and reranker models...")
            from src.core.services.grounding_service import _get_shared_embedding_model
            _ = _get_shared_embedding_model()
            from src.core.reranker import _get_flashrank_reranker
            _ = _get_flashrank_reranker()
            logger.info("Heavy ML models successfully pre-warmed.")
            
            rag = RAGContextEngine(retriever, pipeline_config, checkpointer=checkpointer)
            logger.info("RAG Engine successfully initialized.")
            try:
                summary = rag.registry.get_registry_summary()
                logger.info(f"Knowledge Registry Summary: Datasets={summary['datasets']}, Domains={summary['domains']}, Total Docs={summary['total_documents_count']}")
            except Exception as reg_err:
                logger.warning(f"Could not load Knowledge Registry summary on startup: {reg_err}")
            
            yield
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        traceback.print_exc()
        yield
    finally:
        if rag:
            logger.info("Shutting down RAG Engine...")
            rag.close()
        try:
            asyncio.get_event_loop().create_task(close_aiohttp_session())
        except Exception:
            pass


# Initialize FastAPI App
app = FastAPI(
    title="Modular RAG API Prototype",
    description="A Context Engine utilizing lightweight local embeddings (all-MiniLM-L6-v2) for rapid prototyping, featuring Neural Reranking and SQLite-based Persistent Memory.",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

from starlette.types import ASGIApp, Scope, Receive, Send

class ConditionalGZipMiddleware:
    """
    Middleware that wraps GZipMiddleware but bypasses it for streaming endpoints.
    This prevents buffering and chunked encoding errors for Server-Sent Events (SSE).
    """
    def __init__(self, app: ASGIApp, minimum_size: int = 1024):
        self.app = app
        self.gzip_middleware = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/query_stream", "/resume_stream") or "_stream" in path:
                await self.app(scope, receive, send)
                return
        await self.gzip_middleware(scope, receive, send)

# Enable Gzip compression for payloads > 1024 bytes, bypassing streaming endpoints
app.add_middleware(ConditionalGZipMiddleware, minimum_size=1024)

# Custom static files class with caching headers
class CachedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith((".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2")):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"
        return response

# Serve static assets (CSS, JS) with caching headers
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", CachedStaticFiles(directory=static_dir), name="static")


# Pure ASGI middleware to set session_id context variable for logging.
# IMPORTANT: This must NOT use @app.middleware("http") / BaseHTTPMiddleware because
# BaseHTTPMiddleware wraps StreamingResponse bodies through an intermediate
# anyio MemoryObjectStream, which causes premature stream closure and
# ERR_INCOMPLETE_CHUNKED_ENCODING errors on SSE endpoints.
class SessionLoggingMiddleware:
    """Pure ASGI middleware that extracts session_id and sets the logging context variable."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        session_id = None
        path = scope.get("path", "")

        # 1. Parse session_id from URL path segments
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("history", "sessions", "pending_approval", "resume_stream"):
            session_id = parts[1]

        # 2. Check query parameters
        if not session_id:
            from urllib.parse import parse_qs
            qs = scope.get("query_string", b"").decode("utf-8", errors="replace")
            params = parse_qs(qs)
            if "session_id" in params:
                session_id = params["session_id"][0]

        # 3. Check JSON request body for POST requests
        if not session_id and scope.get("method") == "POST":
            content_type = ""
            for hdr_name, hdr_value in scope.get("headers", []):
                if hdr_name.lower() == b"content-type":
                    content_type = hdr_value.decode("utf-8", errors="replace")
                    break

            if "application/json" in content_type:
                # Buffer the raw body bytes from the ASGI receive channel
                body_chunks = []
                while True:
                    message = await receive()
                    body_chunks.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
                body = b"".join(body_chunks)

                try:
                    if body:
                        data = json.loads(body)
                        session_id = data.get("session_id")
                except Exception:
                    pass

                # Replace receive so downstream handlers can still read the body
                body_replayed = False
                async def replay_receive():
                    nonlocal body_replayed
                    if not body_replayed:
                        body_replayed = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    # Subsequent calls return empty (Starlette caches after first read)
                    return {"type": "http.request", "body": b"", "more_body": False}

                receive = replay_receive

        token = session_id_var.set(session_id)
        try:
            await self.app(scope, receive, send)
        finally:
            session_id_var.reset(token)

app.add_middleware(SessionLoggingMiddleware)


# ------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    question: str
    session_id: str = "default"
    mode: Literal["context_engine", "normal", "agentic"] = "context_engine"
    source_filter: Optional[str] = None
    context_limit: Optional[int] = None
    bypass_hitl: bool = False


class CreateSessionRequest(BaseModel):
    """Request schema for creating a new session."""
    session_id: Optional[str] = None
    title: str = "New Chat"


class RenameSessionRequest(BaseModel):
    """Request schema for renaming a session."""
    title: str


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def serve_ui():
    """Serves the main UI dashboard."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)


@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Primary endpoint for AI chat.
    Orchestrates the retrieval and generation pipeline.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")

    return await rag.ask_async(
        request.question,
        session_id=request.session_id,
        mode=request.mode,
        source_filter=request.source_filter,
        top_k=5,
        context_limit=request.context_limit,
        bypass_hitl=request.bypass_hitl
    )


@app.post("/query_stream")
async def query_rag_stream(request: QueryRequest):
    """
    Streaming endpoint for AI chat.
    Orchestrates the retrieval and generation pipeline chunk-by-chunk.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    
    async def event_generator():
        try:
            logger.info(f"Stream Query from session {request.session_id}: {request.question[:50]}...")
            async for event in rag.ask_stream_async(
                request.question,
                session_id=request.session_id,
                mode=request.mode,
                source_filter=request.source_filter,
                context_limit=request.context_limit,
                bypass_hitl=request.bypass_hitl
            ):
                yield f"data: {_serialize_event(event)}\n\n"
        except asyncio.CancelledError:
            logger.warning(f"Client disconnected from query stream for session {request.session_id}.")
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"data: {_serialize_event({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Handles file uploads, extracts text, and indexes it into the vector store.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    try:
        filename = file.filename
        logger.info(f"Processing upload for: {filename}")

        content = ""
        if filename.endswith(".txt"):
            content = (await file.read()).decode("utf-8")
        elif filename.endswith(".pdf"):
            pdf_data = await file.read()
            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        if not content.strip():
            raise HTTPException(status_code=400, detail="File content is empty.")

        splitter = RecursiveCharacterSplitter(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(content)

        logger.info(f"Generated {len(chunks)} semantic chunks. Indexing...")
        await asyncio.to_thread(retriever.add_documents, chunks, source=filename)

        return {"status": "success", "message": f"Indexed {len(chunks)} chunks from {filename}"}
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Returns system performance and database statistics."""
    if not rag:
        return {"status": "initializing"}
    return {
        "queries_handled": rag.stats["queries"],
        "avg_compression": round(rag.stats["avg_compression_ratio"], 3),
        "avg_latency_ms": round(rag.stats.get("avg_latency_ms", 0.0), 2),
        "document_count": retriever.get_count(),
        "cpu_usage_percent": psutil.cpu_percent(interval=None),
        "memory_usage_percent": psutil.virtual_memory().percent,
        "status": "online"
    }


@app.get("/history/{session_id}")
async def get_session_history(session_id: str):
    """Retrieves conversation history for a specific session."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    return rag.persistent_memory.get_history(session_id)


# ------------------------------------------------------------------
# Session Management Endpoints
# ------------------------------------------------------------------

@app.get("/sessions")
async def list_sessions():
    """Returns all sessions ordered by most recently updated."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    return rag.persistent_memory.list_sessions()


@app.post("/sessions", status_code=201)
async def create_session(request: CreateSessionRequest):
    """Creates a new chat session."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    sid = request.session_id or ("SID-" + uuid.uuid4().hex[:8].upper())
    rag.persistent_memory.create_session(sid, request.title)
    return {"session_id": sid, "title": request.title}


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, request: RenameSessionRequest):
    """Renames an existing session."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    rag.persistent_memory.rename_session(session_id, request.title.strip()[:80])
    return {"session_id": session_id, "title": request.title}


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Deletes a session and all its conversation history."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    rag.delete_session(session_id)
    return None



# ------------------------------------------------------------------
# Approval Management Endpoints (Human-in-the-Loop)
# ------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    session_id: str
    approve: bool

@app.get("/pending_approval/{session_id}")
async def get_pending_approval(session_id: str):
    """Check if there are pending file changes awaiting approval."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    from src.agents.coding_worker import get_pending_approval
    pending = get_pending_approval(session_id)
    if pending:
        return {"has_pending": True, "filepath": pending["filepath"], "tool": pending["tool"]}
    return {"has_pending": False}

@app.post("/approve_changes")
async def approve_changes(request: ApprovalRequest):
    """Approve or reject pending file changes and continue workflow."""
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    from src.agents.coding_worker import get_pending_approval, execute_pending_approval, clear_pending_approval
    from langchain_core.messages import HumanMessage, AIMessage
    from src.graph.workflow import get_graph_config
    from src.graph.supervisor import approve_file
    from src.tools.coding_tools import _get_absolute_path
    
    session_id = request.session_id
    pending = get_pending_approval(session_id)
    
    if not pending:
        return {"status": "no_pending_approval"}
    
    config = get_graph_config(session_id)
    
    if request.approve:
        try:
            abs_path = os.path.realpath(_get_absolute_path(pending['filepath']))
            approve_file(session_id, abs_path)
        except Exception as e:
            logger.warning(f"Failed to register approval for {pending['filepath']}: {e}")
            
        result = execute_pending_approval(session_id)
        logger.info(f"Approval executed: {result}")
        
        if hasattr(rag.multi_agent_graph, "aupdate_state"):
            await rag.multi_agent_graph.aupdate_state(config, {
                "scratchpad": f"\n- [SYSTEM HITL]: User approved modifications. Action result: {result}",
                "waiting_for_approval": False,
                "pending_file_approvals": {},
                "approval_decision": "approved",
                "next_agent": "supervisor",
                "current_task": "Resume after approved file operation.",
                "coding_worker_resume_tool_result": result,
                "coding_worker_resume_tool_call_id": pending.get("tool_call_id")
            })
        else:
            rag.multi_agent_graph.update_state(config, {
                "scratchpad": f"\n- [SYSTEM HITL]: User approved modifications. Action result: {result}",
                "waiting_for_approval": False,
                "pending_file_approvals": {},
                "approval_decision": "approved",
                "next_agent": "supervisor",
                "current_task": "Resume after approved file operation.",
                "coding_worker_resume_tool_result": result,
                "coding_worker_resume_tool_call_id": pending.get("tool_call_id")
            })
        
        return {"status": "approved", "result": result, "message": "Tool executed successfully"}
    else:
        clear_pending_approval(session_id)
        try:
            if hasattr(rag.multi_agent_graph, "aupdate_state"):
                await rag.multi_agent_graph.aupdate_state(config, {
                    "waiting_for_approval": False,
                    "pending_file_approvals": {},
                    "worker_outputs": {"coding_worker": "User rejected file changes."},
                    "approval_decision": "rejected",
                    "next_agent": "supervisor",
                    "current_task": "User rejected file changes; continue without applying them.",
                    "coding_worker_resume_tool_result": "Error: User rejected the file modifications.",
                    "coding_worker_resume_tool_call_id": pending.get("tool_call_id")
                })
            else:
                rag.multi_agent_graph.update_state(config, {
                    "waiting_for_approval": False,
                    "pending_file_approvals": {},
                    "worker_outputs": {"coding_worker": "User rejected file changes."},
                    "approval_decision": "rejected",
                    "next_agent": "supervisor",
                    "current_task": "User rejected file changes; continue without applying them.",
                    "coding_worker_resume_tool_result": "Error: User rejected the file modifications.",
                    "coding_worker_resume_tool_call_id": pending.get("tool_call_id")
                })
        except Exception:
            pass
        return {"status": "rejected"}


@app.get("/resume_stream/{session_id}")
async def resume_stream(session_id: str):
    """
    Resume a paused workflow (e.g., after file approval) and stream the continued
    execution as Server-Sent Events.  The frontend calls this after POSTing to
    /approve_changes so the user can see the resumed agent output live.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")

    from src.graph.workflow import get_graph_config

    async def event_generator():
        config = get_graph_config(session_id)
        try:
            logger.info(f"Resuming workflow stream for session {session_id}")
            if hasattr(rag.multi_agent_graph, "astream"):
                async for event in rag.multi_agent_graph.astream(None, config=config):
                    for node_name, state_delta in event.items():
                        yield f"data: {_serialize_event({'event': 'node_start', 'node': node_name})}\n\n"

                        worker_type = state_delta.get("worker_type", "") or node_name.replace("_node", "")
                        response = ""
                        if "worker_outputs" in state_delta and worker_type:
                            response = state_delta["worker_outputs"].get(worker_type, "")
                        if not response and "messages" in state_delta and state_delta["messages"]:
                            response = state_delta["messages"][-1].content

                        if response:
                            yield f"data: {_serialize_event({'event': 'thought', 'text': f'{worker_type}: {response[:120]}'})}\n\n"
                            yield f"data: {_serialize_event({'event': 'observation', 'output': response})}\n\n"

                        if state_delta.get("waiting_for_approval"):
                            approval_filepath = state_delta.get("approval_filepath", "")
                            approval_tool = state_delta.get("approval_tool", "")
                            yield f"data: {_serialize_event({'event': 'blocked_tool', 'filepath': approval_filepath, 'tool': approval_tool})}\n\n"
                            yield f"data: {_serialize_event({'event': 'waiting_for_approval', 'filepath': approval_filepath, 'tool': approval_tool})}\n\n"
                            return

                        if "final_answer" in state_delta and state_delta["final_answer"]:
                            answer = state_delta["final_answer"]
                            for chunk in answer.split(" "):
                                yield f"data: {_serialize_event({'event': 'answer_chunk', 'text': chunk + ' '})}\n\n"

                yield f"data: {_serialize_event({'event': 'done', 'stats': {}})}\n\n"
                
                # Check if workflow is waiting for approval after graph completes
                try:
                    current_state = await rag.multi_agent_graph.aget_state(config)
                    if current_state and current_state.values:
                        state_values = current_state.values
                        waiting_for_approval = state_values.get("waiting_for_approval", False)
                        if waiting_for_approval:
                            pending_file_approvals = state_values.get("pending_file_approvals", {})
                            approval_filepath = state_values.get("approval_filepath", "")
                            approval_tool = state_values.get("approval_tool", "")
                            
                            approval_packet = {
                                "waiting_for_approval": True,
                                "pending_file_approvals": pending_file_approvals,
                                "approval_filepath": approval_filepath,
                                "approval_tool": approval_tool,
                                "text": "\n\n⚠️ **Action Required:** This modification requires security validation. Please approve or reject below."
                            }
                            yield f"data: {_serialize_event(approval_packet)}\n\n"
                except Exception as state_err:
                    logger.warning(f"Failed to get state after stream: {state_err}")
        except asyncio.CancelledError:
            logger.warning(f"Client disconnected from resume stream for session {session_id}.")
        except Exception as e:
            logger.error(f"Resume stream error: {e}")
            yield f"data: {_serialize_event({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    logger.info("Launching Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
