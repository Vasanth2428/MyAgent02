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
import logging
import traceback
import psutil
import asyncio
import sys
import sentence_transformers
import transformers

print(f"DEBUG: sys.executable = {sys.executable}", flush=True)
print(f"DEBUG: sentence-transformers = {sentence_transformers.__version__}", flush=True)
print(f"DEBUG: transformers = {transformers.__version__}", flush=True)
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pypdf import PdfReader
from dotenv import load_dotenv
import uuid

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP, PipelineConfig
from src.core.retriever import WeaviateRetriever
from src.core.engine import RAGContextEngine
from src.core.splitter import RecursiveCharacterSplitter
from src.core.scraper import close_aiohttp_session

# Load environment variables
load_dotenv()

# Configure Application Logging
import logging.handlers
os.makedirs("logs", exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    "logs/rag_engine.log", maxBytes=5*1024*1024, backupCount=3
)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

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
        
        # Setup safe checkpoints folder path
        import os
        safe_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(safe_dir, exist_ok=True)
        db_path = os.getenv("CHECKPOINTER_DB_PATH", "checkpoints.db")
        from src.graph.checkpointer import validate_db_path
        db_path = validate_db_path(db_path)
        full_path = os.path.join(safe_dir, db_path)
        
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        
        logger.info(f"Using AsyncSqliteSaver with database: {full_path}")
        async with AsyncSqliteSaver.from_conn_string(full_path) as checkpointer:
            await checkpointer.setup()
            
            # Pre-warm heavy ML models to eliminate first-query latency spikes
            logger.info("Pre-warming local embedding and reranker models...")
            _ = retriever.embedding_model
            from src.core.reranker import _get_cross_encoder
            _ = _get_cross_encoder()
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

# Enable Gzip compression for payloads > 1024 bytes
app.add_middleware(GZipMiddleware, minimum_size=1024)

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
    try:
        logger.info(f"Query from session {request.session_id}: {request.question[:50]}...")
        # Keep session's updated_at fresh
        rag.persistent_memory.touch_session(request.session_id)
        result = await rag.ask_async(
            request.question,
            session_id=request.session_id,
            mode=request.mode,
            source_filter=request.source_filter,
            context_limit=request.context_limit
        )
        return result
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error processing query: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}\n\nTRACEBACK:\n{error_details}")


@app.post("/query_stream")
async def query_rag_stream(request: QueryRequest):
    """
    Streaming endpoint for AI chat.
    Orchestrates the retrieval and generation pipeline chunk-by-chunk.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="Engine not ready")
    
    async def event_generator():
        import json
        try:
            logger.info(f"Stream Query from session {request.session_id}: {request.question[:50]}...")
            async for event in rag.ask_stream_async(
                request.question,
                session_id=request.session_id,
                mode=request.mode,
                source_filter=request.source_filter,
                context_limit=request.context_limit
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            logger.warning(f"Client disconnected from query stream for session {request.session_id}.")
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

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


if __name__ == "__main__":
    import uvicorn
    logger.info("Launching Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
