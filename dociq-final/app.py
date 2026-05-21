import os
import gc
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.pdf_processor import process_pdf, safe_clear
from utils.rag_chain import build_rag_chain, retrieve_chunks

st.set_page_config(
    page_title="DocIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600,700&display=swap');

:root {
    --bg:         #0d0f12;
    --surface:    #13161b;
    --surface-2:  #181c22;
    --border:     rgba(255,255,255,0.07);
    --text:       #e2e4e9;
    --text-muted: #6b7280;
    --text-faint: #374151;
    --accent:     #4f8ef7;
    --accent-dim: rgba(79,142,247,0.12);
    --green:      #34d399;
    --green-dim:  rgba(52,211,153,0.10);
    --red:        #f87171;
    --radius:     10px;
    --radius-sm:  6px;
    --ease:       180ms cubic-bezier(0.16,1,0.3,1);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Satoshi', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg); color: var(--text);
}
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.25rem; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stButton > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-family: 'Satoshi', sans-serif !important; font-weight: 600 !important;
    font-size: 0.85rem !important; padding: 0.5rem 1rem !important;
    transition: opacity var(--ease) !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important; color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    color: var(--text) !important;
    border-color: rgba(255,255,255,0.15) !important; opacity: 1 !important;
}
.stTextInput input, .stSelectbox > div > div {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text) !important;
    font-family: 'Satoshi', sans-serif !important; font-size: 0.9rem !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
.msg-user {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) 3px var(--radius);
    padding: 0.875rem 1.125rem; margin: 0.75rem 0;
    font-size: 0.925rem; line-height: 1.6; color: var(--text);
}
.msg-assistant {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    border-radius: 3px var(--radius) var(--radius) var(--radius);
    padding: 1rem 1.25rem; margin: 0.75rem 0;
    font-size: 0.925rem; line-height: 1.75; color: var(--text);
}
.badge-online {
    display:inline-flex; align-items:center; gap:6px;
    background:var(--green-dim); border:1px solid rgba(52,211,153,0.2);
    border-radius:999px; padding:4px 12px;
    font-size:0.78rem; font-weight:600; color:var(--green); letter-spacing:0.02em;
}
.badge-offline {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(248,113,113,0.1); border:1px solid rgba(248,113,113,0.2);
    border-radius:999px; padding:4px 12px;
    font-size:0.78rem; font-weight:600; color:var(--red);
}
.dot { width:6px; height:6px; border-radius:50%; background:currentColor; }
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
.stat-card {
    background:var(--surface-2); border:1px solid var(--border);
    border-radius:var(--radius-sm); padding:10px 12px; text-align:center;
}
.stat-val { font-size:1.4rem; font-weight:700; color:var(--accent); line-height:1.2; }
.stat-lbl { font-size:0.72rem; color:var(--text-muted); margin-top:2px;
            letter-spacing:0.04em; text-transform:uppercase; }
.section-label {
    font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
    text-transform:uppercase; color:var(--text-faint);
    margin-bottom:8px; margin-top:20px;
}
.divider { height:1px; background:var(--border); margin:16px 0; }
.streamlit-expanderHeader {
    font-size:0.82rem !important; color:var(--text-muted) !important;
    background:var(--surface-2) !important; border-radius:var(--radius-sm) !important;
}
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
</style>
""", unsafe_allow_html=True)


def ollama_online():
    try:
        return requests.get("http://localhost:11434", timeout=2).status_code == 200
    except Exception:
        return False


def get_ollama_models():
    try:
        data = requests.get("http://localhost:11434/api/tags", timeout=2).json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


for key, default in {
    "history": [], "vectorstore": None, "session_dir": None,
    "processed": False, "doc_name": None, "chunk_count": 0,
    "llm_model": "llama3.2", "embed_model": "nomic-embed-text",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:24px">
        <div style="font-size:1.1rem;font-weight:700;color:#e2e4e9;letter-spacing:-0.01em">DocIQ</div>
        <div style="font-size:0.78rem;color:#6b7280;margin-top:2px">Document Intelligence System</div>
    </div>""", unsafe_allow_html=True)

    if ollama_online():
        st.markdown('<div class="badge-online"><span class="dot"></span>Ollama running</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-offline"><span class="dot"></span>Ollama offline</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:12px;font-size:0.82rem;color:#6b7280">Start Ollama to continue:</div>', unsafe_allow_html=True)
        st.code("ollama serve", language="bash")
        st.stop()

    all_models   = get_ollama_models()
    llm_models   = [m for m in all_models if "embed" not in m.lower()] or ["llama3.2"]
    embed_models = [m for m in all_models if "embed" in m.lower()] or ["nomic-embed-text"]

    def idx(lst, val):
        return lst.index(val) if val in lst else 0

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)

    selected_llm = st.selectbox(
        "Generation", llm_models,
        index=idx(llm_models, st.session_state.llm_model),
        label_visibility="collapsed",
        help="LLM used for answer generation",
    )
    selected_embed = st.selectbox(
        "Embedding", embed_models,
        index=idx(embed_models, st.session_state.embed_model),
        label_visibility="collapsed",
        help="Model used to vectorise PDF chunks",
    )
    st.session_state.llm_model   = selected_llm
    st.session_state.embed_model = selected_embed

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
    top_k    = st.slider("Retrieved chunks", 3, 10, 5)

    c1, c2 = st.columns(2)
    process_btn = c1.button("Process", type="primary",   use_container_width=True)
    clear_btn   = c2.button("Clear",   type="secondary", use_container_width=True)

    if process_btn:
        if not uploaded:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Processing..."):
                prev_vs  = st.session_state.vectorstore
                prev_dir = st.session_state.session_dir
                st.session_state.vectorstore = None
                gc.collect()

                if prev_vs is not None and prev_dir:
                    safe_clear(prev_vs, prev_dir)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                try:
                    n, vs, sdir = process_pdf(tmp_path, selected_embed)
                    st.session_state.vectorstore = vs
                    st.session_state.session_dir = sdir
                    st.session_state.processed   = True
                    st.session_state.doc_name    = uploaded.name
                    st.session_state.chunk_count = n
                    st.session_state.history     = []
                    st.success(f"{n} chunks indexed")
                except Exception as e:
                    st.error(str(e))

    if clear_btn:
        prev_vs  = st.session_state.vectorstore
        prev_dir = st.session_state.session_dir
        st.session_state.vectorstore = None
        st.session_state.session_dir = None
        gc.collect()
        if prev_vs is not None and prev_dir:
            safe_clear(prev_vs, prev_dir)
        st.session_state.processed   = False
        st.session_state.doc_name    = None
        st.session_state.chunk_count = 0
        st.session_state.history     = []
        st.rerun()

    if st.session_state.processed:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-val">{st.session_state.chunk_count}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{top_k}</div>
                <div class="stat-lbl">Top-K</div>
            </div>
        </div>
        <div style="margin-top:10px;font-size:0.78rem;color:#6b7280;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
            {st.session_state.doc_name}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("Setup guide"):
        st.markdown("""
**First-time setup**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
pip install -r requirements.txt
streamlit run app.py
```
**Windows:** Ollama auto-starts on boot — do not run `ollama serve` manually.
        """)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.06)">
    <h1 style="font-size:1.5rem;font-weight:700;letter-spacing:-0.02em;color:#e2e4e9;margin-bottom:4px">
        Document Intelligence
    </h1>
    <p style="font-size:0.875rem;color:#6b7280;margin:0">
        Ask questions about any PDF — answers grounded in your document, fully local.
    </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.processed:
    st.markdown("""
    <div style="max-width:560px;margin-top:40px">
        <div style="font-size:0.9rem;color:#6b7280;line-height:1.8">
            Upload a PDF in the sidebar and click <strong style="color:#e2e4e9">Process</strong> to begin.
            <br><br>
            Works with research papers, compliance documents, SOPs, contracts, and resumes.
            All processing happens locally — no data leaves your machine.
        </div>
    </div>""", unsafe_allow_html=True)

else:
    for msg in st.session_state.history:
        css = "msg-user" if msg["role"] == "user" else "msg-assistant"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"Sources — {len(msg['sources'])} chunks retrieved"):
                for i, chunk in enumerate(msg["sources"]):
                    fname   = chunk.metadata.get("source_file", "unknown")
                    page    = chunk.metadata.get("page", "?")
                    preview = chunk.page_content[:350]
                    if len(chunk.page_content) > 350:
                        preview += "..."
                    st.markdown(f"**Chunk {i+1}** &nbsp;·&nbsp; `{fname}` &nbsp;·&nbsp; page {page}")
                    st.code(preview, language=None)
                    if i < len(msg["sources"]) - 1:
                        st.markdown("---")

    if not st.session_state.history:
        suggestions = [
            "Summarise this document in 5 bullet points",
            "What are the key findings or conclusions?",
            "List all important dates and deadlines",
            "What are the main risks or limitations?",
        ]
        st.markdown('<div class="section-label" style="margin-top:0">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, key=f"chip_{i}", use_container_width=True, type="secondary"):
                st.session_state._pending = q
                st.rerun()

    pending = st.session_state.pop("_pending", None)

    with st.form("input_form", clear_on_submit=True):
        col_q, col_btn = st.columns([6, 1])
        with col_q:
            user_input = st.text_input(
                "question", value=pending or "",
                placeholder="Ask a question about your document...",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.form_submit_button("Send", type="primary", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()
        st.session_state.history.append({"role": "user", "content": question})
        with st.spinner("Generating answer..."):
            try:
                chain, retriever = build_rag_chain(
                    st.session_state.vectorstore, top_k, selected_llm
                )
                answer  = chain.invoke(question)
                sources = retrieve_chunks(retriever, question)
                st.session_state.history.append({
                    "role": "assistant", "content": answer, "sources": sources,
                })
            except Exception as e:
                st.session_state.history.append({
                    "role": "assistant",
                    "content": f"Error generating response: {str(e)}",
                })
        st.rerun()

    if st.session_state.history:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.history = []
            st.rerun()
