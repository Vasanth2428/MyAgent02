"""
================================================================================
RAG CONTEXT ENGINE API (FastAPI)
================================================================================
This is the primary entry point for the RAG system. It exposes REST endpoints for:
- Document Upload (.pdf, .txt)
- Semantic Search & Querying (Dual-mode)
- System Monitoring (Stats)
- Conversation History
- Static File Serving (UI)
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
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional
from pypdf import PdfReader
from dotenv import load_dotenv

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.retriever import WeaviateRetriever
from core.engine import RAGContextEngine
from core.splitter import RecursiveCharacterSplitter

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
        rag = RAGContextEngine(retriever)
        logger.info("RAG Engine successfully initialized.")
        yield
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        traceback.print_exc()
        yield
    finally:
        if rag:
            logger.info("Shutting down RAG Engine...")
            rag.close()


# Initialize FastAPI App
app = FastAPI(
    title="Premium Modular RAG API",
    description="A high-performance Context Engine with Neural Reranking and Persistent Memory.",
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

# Serve static assets (CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    question: str
    session_id: str = "default"
    mode: Literal["context_engine", "normal", "agentic"] = "context_engine"
    source_filter: Optional[str] = None


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

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
        result = await asyncio.to_thread(
            rag.ask,
            request.question,
            session_id=request.session_id,
            mode=request.mode,
            source_filter=request.source_filter
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
            loop = asyncio.get_event_loop()
            
            def run_sync_gen(queue):
                try:
                    for event in rag.ask_stream(
                        request.question,
                        session_id=request.session_id,
                        mode=request.mode,
                        source_filter=request.source_filter
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "message": str(e)})
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            queue = asyncio.Queue()
            loop.run_in_executor(None, run_sync_gen, queue)

            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, dict) and event.get("event") == "error":
                    yield f"data: {json.dumps(event)}\n\n"
                    break
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


if __name__ == "__main__":
    import uvicorn
    logger.info("Launching Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
