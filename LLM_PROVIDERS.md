# LLM Provider Configuration

The multi-agent system uses `src/core/model_provider.py` as its single model
construction boundary. Workers should not import provider-specific chat classes.

## Supported providers

- `groq`
- `google_genai` (`gemini` and `google` are accepted aliases)
- `openai`
- `cerebras` through its OpenAI-compatible API

Install the dependencies from `config/requirements.txt`, then select a global
provider:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=...
```

## Move only the coding worker to Gemini

```env
LLM_PROVIDER=groq
GROQ_API_KEY=...

CODING_WORKER_LLM_PROVIDER=google_genai
CODING_WORKER_MODEL_PRIMARY=gemini-2.5-flash
CODING_WORKER_MODEL_FALLBACK=gemini-2.5-flash-lite
GOOGLE_API_KEY=...
```

All other workers continue using Groq.

## Use a different fallback provider

```env
CODING_WORKER_LLM_PROVIDER=google_genai
CODING_WORKER_MODEL_PRIMARY=gemini-2.5-flash

CODING_WORKER_LLM_FALLBACK_PROVIDER=cerebras
CODING_WORKER_MODEL_FALLBACK=gpt-oss-120b

GOOGLE_API_KEY=...
CEREBRAS_API_KEY=...
```

## Worker-specific settings

Each role accepts:

```text
<ROLE>_LLM_PROVIDER
<ROLE>_LLM_PRIMARY_PROVIDER
<ROLE>_LLM_FALLBACK_PROVIDER
<ROLE>_MODEL_PRIMARY
<ROLE>_MODEL_FALLBACK
```

Available role prefixes currently include:

```text
SUPERVISOR
RAG_WORKER
CODING_WORKER
CODE_CRITIC
CRITIC
SYNTHESIZER
REPORT_WORKER
WEB_WORKER
SCRAPER_WORKER
UTILITY_WORKER
EXPANDER
HYDE
```

## Remaining legacy boundary

`src/core/llm.py` and `src/core/services/generation_service.py` still expose the
raw Groq/OpenAI-style completion interface used by the core RAG answer pipeline.
The multi-agent system, including the coding worker, no longer depends directly
on `ChatGroq`. Migrating the legacy generation boundary should be a separate
change because it includes synchronous, asynchronous, and streaming response
normalization.
