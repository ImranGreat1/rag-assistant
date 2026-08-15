# Manual RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built without a framework like LangChain. Every stage — document loading, chunking, embedding, retrieval, prompt construction, and inference — is implemented directly against the underlying SDKs (Pinecone, OpenAI-compatible client) rather than through a framework's abstractions.

## Why manual instead of LangChain?

This pipeline exists alongside a LangChain-based implementation in this project (see `pipelines/langchain_pipeline/`). This version favors:

- **Transparency** — every step (chunking, embedding, retrieval, prompt assembly, streaming) is plain, readable Python with no hidden abstractions to trace through.
- **Minimal dependencies** — no LangChain runtime, chains, or retriever abstractions; just the Pinecone and OpenAI SDKs.
- **Full control over streaming and tracing** — the pipeline is hand-wired as an async generator so that streamed LLM output and LangSmith tracing nest correctly under a single trace (see [Tracing](#tracing) below).

Use this pipeline when you want to understand or modify exactly what happens at each stage, or when you want to avoid LangChain's dependency footprint.

## Pipeline Overview

```
Documents (.txt, .pdf)
      │
      ▼
TextFileLoader ──► CharacterTextSplitter ──► chunks
      │
      ▼
Pinecone embeddings (hosted model) ──► upsert to Pinecone index
      │
      ▼
 ┌─────────────────────── runtime request flow ───────────────────────┐
 │                                                                     │
 │  User query (via API)                                              │
 │        │                                                           │
 │        ▼                                                           │
 │  Retrieve relevant chunks from Pinecone                            │
 │        │                                                           │
 │        ▼                                                           │
 │  Construct prompt (system + user templates + retrieved context)    │
 │        │                                                           │
 │        ▼                                                           │
 │  ChatOpenAI (Gemini via OpenAI-compatible endpoint)                │
 │        │                                                           │
 │        ▼                                                           │
 │  Stream or await full response                                     │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
```

Every stage of the runtime flow is instrumented with LangSmith's `@traceable` decorator, producing a single nested trace per request.

## Building Blocks

### 1. Document loading & splitting — `utils/text.py`

| Class | Responsibility |
|---|---|
| `TextFileLoader` | Loads raw text content from `.txt` or `.pdf` source documents. |
| `CharacterTextSplitter` | Splits loaded documents into overlapping/fixed-size chunks suitable for embedding. |

These are plain helper classes with no external dependencies beyond the standard library — used only at ingestion time, not at request time.

### 2. Vector store — `infra/pinecone.py`

A thin helper class wrapping the Pinecone SDK, responsible for:

- Inserting (upserting) embedded chunks into the configured Pinecone index.
- Searching the index for chunks relevant to a given query.

This is the only module in the pipeline that talks to Pinecone directly — both the ingestion script and the runtime retrieval step go through it.

### 3. Ingestion script — `scripts/setup_health_store.py`

A standalone, one-off script (not part of the request-time pipeline) that:

1. Loads source documents via `TextFileLoader`.
2. Splits them into chunks via `CharacterTextSplitter`.
3. Embeds each chunk using Pinecone's hosted open-source embedding model.
4. Upserts the embedded chunks into the Pinecone vector store via the `infra/pinecone.py` helper.

Run this whenever source documents change or the index needs to be (re)built:

```bash
python scripts/setup_health_store.py
```

### 4. LLM client — `chat_openai.py` (`ChatOpenAI` helper class)

Wraps the OpenAI SDK's async client, configured against Gemini's OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`) rather than OpenAI itself. Responsibilities:

- Initializes an `AsyncOpenAI` client, optionally wrapped with `wrap_openai` for automatic LangSmith instrumentation.
- `run(messages, stream)` — performs inference; returns a plain string when `stream=False`, or an async generator of text chunks when `stream=True`.

### 5. Pipeline orchestration — `rag_pipelines/manual/pipeline.py` (`ManualRagPipeline` class)

Ties the building blocks together at request time:

1. **Retrieve** — query Pinecone (via `infra/pinecone.py`) for chunks relevant to the user's query.
2. **Construct prompts** — merge the user query and retrieved context into the system/user prompt templates.
3. **Infer** — call `ChatOpenAI.run(...)` with the constructed prompts.
4. **Respond** — stream chunks back to the caller, or await and return the full response, depending on the `stream` flag.

## Tracing

Every stage of the runtime pipeline — retrieval, prompt construction, and inference — is wrapped with LangSmith's `@traceable` decorator, and the OpenAI client is wrapped with `wrap_openai` for automatic instrumentation of the raw LLM call.

**A note on streaming and trace nesting:** because Python decides whether a function is a coroutine or an async generator at definition time (based on whether it contains `yield`), the pipeline's entry method is itself implemented as an async generator — even for non-streaming requests, it yields a single item. This keeps its `@traceable` span open for the full duration of token generation, so the nested retrieval, prompt-construction, and inference spans all attach correctly under one trace instead of the LLM call spawning a disconnected, standalone trace.

## Configuration

| Env var | Purpose |
|---|---|
| `GEMINI_API_KEY` | API key used by `ChatOpenAI` to authenticate against the Gemini OpenAI-compatible endpoint. |
| `PINECONE_API_KEY` | API key for the Pinecone vector store. |
| *(LangSmith env vars)* | Standard `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, etc., required for `@traceable` to emit traces. |

## Usage

```python
pipeline = ManualRagPipeline()

# Streaming
async for chunk in pipeline.run_pipeline(query="How does REM sleep work?", stream=True):
    print(chunk, end="", flush=True)

# Non-streaming
async for result in pipeline.run_pipeline(query="How does REM sleep work?", stream=False):
    print(result)
```