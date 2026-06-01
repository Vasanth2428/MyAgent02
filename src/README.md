# Multi-Agent System with LangGraph (Supervisor + Workers)

A production-grade multi-agent system using a supervisor pattern for query routing.

## Architecture

```
User Query ? Supervisor ? [RAG Worker | Web Worker | Utility Worker] ? Supervisor ? ... ? Final Answer
```

### Components

- **Supervisor**: Routes queries to appropriate workers using gpt-4o-mini
- **RAG Worker**: Answers STRICTLY from documents (no external knowledge)
- **Web Worker**: Fetches live web data via Tavily
- **Utility Worker**: Handles calculations, datetime, summarization

## Setup

1. Install dependencies:
```bash
pip install -r src/requirements.txt
```

2. Configure environment variables (copy `.env.example` to `.env`):
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: For web search
- `SUPERVISOR_MODEL`: gpt-4o-mini (default)
- `REASONING_MODEL`: gpt-4o (default)

## Usage

### CLI
```python
python -m src.main "What is the weather today?"
python -m src.main "Calculate 15% of 200"
python -m src.main "What does my document say about X?"
```

### Programmatic
```python
from src.main import initialize_graph, run_query
from src.graph.workflow import build_multi_agent_graph

graph = initialize_graph()
result = run_query("Your question here")
```

### Streaming
```python
for event in graph.astream(input_state, config=config):
    print(event)
```

## Observability

Set `LANGCHAIN_TRACING_V2=true` with a LangSmith API key to enable tracing.

## Checkpointing

Uses SqliteSaver by default. Set `USE_SQLITE_CHECKPOINTER=true` and `CHECKPOINTER_DB_PATH` to persist state.
