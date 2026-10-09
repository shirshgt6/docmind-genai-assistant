# DocMind — LangChain RAG & Agent Document Assistant

Upload PDFs, text files or web pages and **chat with them**: answers are grounded in your
documents with **citations**, follow-up questions keep **conversation memory**, a **router**
picks the right pipeline (Q&A / summary / extraction), and a **tool-calling agent** can search
documents, do math and check dates.

Built with **LangChain (LCEL)**, **FAISS**, **FastAPI** and **Streamlit**. Works with **Groq (free)**,
**OpenAI** or fully local **Ollama** — switch with one line in `.env`. 21 offline tests (no API key needed).

```
            ┌──────────── Streamlit UI ────────────┐
            │ upload · chat · summary · insights    │
            └───────────────┬──────────────────────┘
                            │ REST
                    ┌───────▼────────┐
                    │  FastAPI (api) │  validation, upload safety, errors, logging
                    └───────┬────────┘
                    ┌───────▼────────┐
                    │ DocMindService │  memory per session, wires the chains
                    └───────┬────────┘
   ┌──────────────┬─────────┼───────────┬──────────────┬──────────────┐
   ▼              ▼         ▼           ▼              ▼              ▼
 Ingestion     RAG chain  Summarize   Extract       Router         Agent
 loaders →     rewrite →  stuff OR    Pydantic      structured     tool calling
 splitter →    MMR →      map-reduce  structured    intent →       search_documents
 embeddings →  prompt →   (branch)    output        dispatch       calculator
 FAISS         LLM                                                  current_date
```

## Features → LangChain concepts (CampusX GenAI playlist)

| Playlist topic | Where it is used |
|---|---|
| Chat models & embeddings | `docmind/models.py` — provider factory (Groq / OpenAI / Ollama, Ollama or HuggingFace embeddings) |
| Prompts, ChatPromptTemplate, MessagesPlaceholder | `docmind/prompts.py` |
| Structured output (Pydantic, `with_structured_output`) | `docmind/chains/extract.py`, `router.py` |
| Output parsers (`StrOutputParser`, `PydanticOutputParser` fallback) | `docmind/chains/structured.py` |
| Chains — sequential, parallel, conditional | RAG (sequential), map step `.map()` (parallel), `RunnableBranch` (conditional) |
| Runnables — `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch` | `docmind/chains/rag.py`, `summarize.py` |
| Document loaders (PDF, text, web) | `docmind/ingest.py` — `PyPDFLoader`, `TextLoader`, `WebBaseLoader` |
| Text splitters | `RecursiveCharacterTextSplitter` with chunk ids for citations |
| Vector stores | FAISS, persisted to disk — `docmind/vectorstore.py` |
| Retrievers | MMR retriever (relevant **and** diverse chunks) |
| RAG | conversational RAG with question rewriting + source citations |
| Tools & tool calling | `docmind/tools.py` — `@tool`, safe AST calculator, retriever tool |
| Agents | `create_tool_calling_agent` + `AgentExecutor` (max 5 iterations) |
| Memory / chat history | `docmind/memory.py` — per-session history, last 10 turns |

## Quick start

Prerequisites: Python 3.10+, [Ollama](https://ollama.com) running locally (for embeddings), a free
[Groq API key](https://console.groq.com).

```bash
git clone https://github.com/shirshgt6/docmind-genai-assistant.git
cd docmind-genai-assistant
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull nomic-embed-text  # local embedding model (~270 MB)
cp .env.example .env          # put your GROQ_API_KEY

uvicorn docmind.api:app --reload          # terminal 1 → http://localhost:8000/docs (Swagger)
streamlit run ui/streamlit_app.py         # terminal 2 → http://localhost:8501
```

Try it with `data/samples/company_handbook.md`:
- *"How many sick leaves do I get?"* → then follow up *"and annual?"* (memory + question rewriting)
- *"Give me a TL;DR"* in Smart mode → routed to **summarize**
- Agent mode: *"If I take 9 days of annual leave, how many are left and what is 85000 / 36 per month for the laptop?"*

Docker: `docker compose up --build` (API on 8000, UI on 8501).

Run tests (offline, fake LLM + fake embeddings): `pytest -q`

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/documents/upload` | upload PDF/TXT/MD (≤10 MB) → chunk → embed → FAISS |
| POST | `/documents/url` | ingest a web page |
| GET | `/documents` | list indexed sources |
| POST | `/ask` | RAG answer + sources, with session memory |
| POST | `/assistant` | router decides: qa / summarize / extract |
| POST | `/summarize` | stuff or map-reduce summary (per source or all) |
| POST | `/extract` | structured `DocumentInsights` JSON |
| POST | `/agent` | tool-calling agent, returns answer + tool steps |
| DELETE | `/sessions/{id}` | clear chat memory |

## Design decisions

- **Dependency injection of models** — every chain takes the LLM/embeddings as arguments, so the
  provider is a config change and tests run with fake models (no network, no cost, deterministic).
- **Question rewriting only when needed** — a `RunnableBranch` skips the extra LLM call on the first turn.
- **MMR retrieval** — avoids sending 4 near-duplicate chunks; `fetch_k = 5k` candidates, then diversify.
- **Citations** — every chunk carries `source#index`; the prompt forces `[source#chunk]` citations and
  an explicit "not found" answer to reduce hallucination.
- **Stuff vs map-reduce** — short docs go in one call (cheap); long docs are summarized per chunk in
  parallel then combined (fits context window).
- **Structured output with fallback** — native tool/JSON mode when the model supports it, otherwise
  `PydanticOutputParser` + format instructions. Output is always a validated Pydantic object.
- **Safe agent** — allowlisted tools only, calculator uses an AST whitelist (no `eval`), capped at 5 steps.
- **Upload safety** — extension allowlist, 10 MB cap, filename sanitised (no `../` path traversal).

## Roadmap

- Streaming responses in the UI
- Per-user document isolation with metadata filters
- Redis-backed chat history and a managed vector DB (Qdrant / pgvector) for multi-instance deployment
- RAG evaluation (retrieval hit-rate, answer faithfulness) on a labeled question set
