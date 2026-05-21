# DocIQ — Document Intelligence System

RAG-powered Q&A for any PDF. Fully local — no API keys, no internet required.

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain LCEL |
| LLM | Ollama (`llama3.2`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | ChromaDB (local) |
| PDF parsing | PyPDF |

## Quick Start

**1. Install Ollama** — [ollama.com/download](https://ollama.com/download)

**2. Pull models**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**3. Install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**4. Run**
```bash
streamlit run app.py
```

> **Windows note:** Ollama auto-starts on boot. Do NOT run `ollama serve` — it will throw a port conflict error if Ollama is already running.

---

## Project Structure

```
dociq/
├── app.py                  # Streamlit UI
├── utils/
│   ├── pdf_processor.py    # PDF loading, chunking, embedding, ChromaDB
│   └── rag_chain.py        # RAG chain and prompt engineering
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Ingest** — PDF is split into 1000-token chunks (150-token overlap), embedded with `nomic-embed-text`, stored in ChromaDB
2. **Retrieve** — Question is embedded, MMR retrieval fetches the top-K most relevant and diverse chunks
3. **Generate** — System prompt + few-shot examples + chain-of-thought + retrieved chunks are passed to `llama3.2`
4. **Respond** — Answer displayed with source citations and expandable chunk viewer

---

## Windows File-Lock Fix

Each PDF upload creates a **unique ChromaDB subdirectory** (`chroma_db/<uuid>/`).
This avoids the `[WinError 32]` file-in-use error caused by Windows holding locks
on active ChromaDB files when a folder deletion is attempted.

On new upload: old vectorstore reference is released → GC is forced → old folder
is deleted with retries → new unique folder is created.

---

## Configuration

| Parameter | Default | Notes |
|---|---|---|
| `chunk_size` | 1000 tokens | `pdf_processor.py` |
| `chunk_overlap` | 150 tokens | Prevents context loss at boundaries |
| `top_k` | 5 | UI slider (3–10) |
| `temperature` | 0.2 | Low for factual accuracy |
| LLM model | `llama3.2` | Auto-detected from installed Ollama models |
| Embed model | `nomic-embed-text` | Auto-detected from installed Ollama models |

---

## Recommended Models

```bash
# LLMs
ollama pull llama3.2          # fast, good quality (default)
ollama pull llama3.1:8b       # better reasoning
ollama pull mistral           # good for structured documents

# Embedding
ollama pull nomic-embed-text  # 768-dim, recommended
ollama pull mxbai-embed-large # 1024-dim, higher quality
```

---

## License

MIT
