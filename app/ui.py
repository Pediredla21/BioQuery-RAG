"""
ui.py — Streamlit UI for BioQuery: Biology Research Paper RAG Assistant

Flow:
  1. Upload PDF(s) in the sidebar.
  2. Click "Build Index" — the app processes and indexes the PDFs.
  3. Ask any question in the chat.
     - If you ask for a summary ("summarize this paper", "give me an overview"),
       the app generates a structured summary automatically.
     - Otherwise, it retrieves relevant passages and answers with citations.
"""

import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make sure the app/ folder is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_pdf_dir, get_vectorstore_dir
from ingest import ingest_pdfs_from_paths
from query import load_vectorstore, answer_question, summarize_paper

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioQuery",
    page_icon="🧬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f, #0e7490);
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .app-header h2 { margin: 0; color: white; font-size: 1.6rem; }
    .app-header p  { margin: 0.3rem 0 0; color: #bae6fd; font-size: 0.9rem; }

    /* Evidence panel card */
    .evidence-card {
        background: #1e293b;
        border-left: 3px solid #0e7490;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .evidence-label {
        font-weight: 600;
        color: #7dd3fc;
        margin-bottom: 0.3rem;
    }

    /* Confidence badges */
    .badge-high   { color: #4ade80; font-weight: 600; }
    .badge-medium { color: #facc15; font-weight: 600; }
    .badge-low    { color: #f87171; font-weight: 600; }

    /* Status step labels in sidebar */
    .step-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def vectorstore_is_ready() -> bool:
    vs = get_vectorstore_dir()
    return (vs / "index.faiss").exists() and (vs / "index.pkl").exists()

def is_summary_request(text: str) -> bool:
    """
    Checks if the user's message is asking for a paper summary.
    Simple keyword check — no ML needed.
    """
    keywords = [
        "summarize", "summarise", "summary", "summarization",
        "overview", "brief me", "what is this paper about",
        "what's this paper about", "give me an overview",
        "explain this paper", "describe this paper",
    ]
    lower = text.lower().strip()
    return any(kw in lower for kw in keywords)

# ── Session state ──────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": str, "content": str}

if "last_citations" not in st.session_state:
    st.session_state.last_citations = [] # citation dicts from the latest answer

if "index_built" not in st.session_state:
    st.session_state.index_built = vectorstore_is_ready()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧬 BioQuery")
    st.caption("Biology Research Paper Assistant")
    st.divider()

    # ── Step 1: Upload ─────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">Step 1 — Upload PDFs</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload one or more biology PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("💾 Save PDFs", use_container_width=True):
            pdf_dir = get_pdf_dir()
            pdf_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                (pdf_dir / f.name).write_bytes(f.read())
            st.success(f"Saved {len(uploaded_files)} file(s).")
            st.rerun()

    # Show list of available PDFs
    pdf_dir = get_pdf_dir()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    available_pdfs = sorted([p.name for p in pdf_dir.glob("*.pdf")])

    if available_pdfs:
        st.markdown("**Uploaded papers:**")
        for name in available_pdfs:
            st.markdown(f"• {name}")
    else:
        st.info("No PDFs yet. Upload above.")

    st.divider()

    # ── Step 2: Build Index ────────────────────────────────────────────────────
    st.markdown('<p class="step-label">Step 2 — Build Index</p>', unsafe_allow_html=True)
    st.caption("Converts your PDFs into a searchable vector database.")

    if st.button("🔨 Build / Rebuild Index", use_container_width=True, type="primary"):
        if not available_pdfs:
            st.error("Upload at least one PDF first.")
        else:
            pdf_paths = [str(pdf_dir / n) for n in available_pdfs]
            with st.spinner("Indexing PDFs… (first run may take ~1 min)"):
                ok, msg = ingest_pdfs_from_paths(pdf_paths)
            if ok:
                st.success(msg)
                st.session_state.index_built = True
                st.session_state.chat_history = []
                st.session_state.last_citations = []
            else:
                st.error(msg)

    status = "✅ Ready" if st.session_state.index_built else "⚠️ Not built yet"
    st.markdown(f"**Index status:** {status}")

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">Settings</p>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="70b = more accurate | 8b = faster",
    )
    top_k = st.slider("Passages to retrieve", min_value=2, max_value=8, value=4)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_citations = []
        st.rerun()

# ── Guard: index not ready ─────────────────────────────────────────────────────

if not st.session_state.index_built:
    st.markdown("""
    <div class="app-header">
        <h2>🧬 Welcome to BioQuery</h2>
        <p>Your biology research paper assistant — ask questions, get cited answers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "**To get started:**\n\n"
        "1. Upload your biology PDF(s) in the sidebar.\n"
        "2. Click **Save PDFs**.\n"
        "3. Click **Build / Rebuild Index**.\n"
        "4. Then come back here to ask questions!"
    )
    st.stop()

# ── Main layout ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h2>🧬 BioQuery</h2>
    <p>Ask questions about your uploaded biology paper(s). Try: <em>"Summarize this paper"</em>, <em>"What methods were used?"</em>, <em>"What are the main findings?"</em></p>
</div>
""", unsafe_allow_html=True)

chat_col, evidence_col = st.columns([3, 2])

# ── LEFT: Chat ─────────────────────────────────────────────────────────────────
with chat_col:
    # Quick-start buttons
    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    sample_questions = [
        "Summarize this paper",
        "What methods are used?",
        "What are the main findings?",
        "What datasets were used?",
        "What are the conclusions?",
        "What is the research question?",
    ]
    for i, q in enumerate(sample_questions):
        if qcols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("---")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Get user input (typed or from quick button)
    prefill = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Ask a question about your paper(s)…")
    question = prefill or user_input

    if question:
        # Show the user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            db = load_vectorstore()

            if is_summary_request(question):
                # ── Summary path ──────────────────────────────────────────────
                with st.spinner("Generating paper summary…"):
                    result = summarize_paper(db, model=selected_model)

                st.markdown(result["summary"])
                st.session_state.last_citations = result["citations"]
                answer_text = result["summary"]

            else:
                # ── Q&A path ──────────────────────────────────────────────────
                with st.spinner("Searching the paper and generating answer…"):
                    result = answer_question(db, question, k=top_k, model=selected_model)

                # Confidence badge
                conf = result["confidence_label"]
                badge = result["confidence_emoji"]
                badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(conf, "")
                st.markdown(
                    f"<span class='{badge_class}'>{badge} {conf} confidence</span>",
                    unsafe_allow_html=True,
                )

                st.markdown(result["answer"])

                # Compact source list below the answer
                unique_sources = sorted({c["label"] for c in result["citations"]})
                if unique_sources:
                    st.caption("📎 Sources: " + " · ".join(unique_sources))

                st.session_state.last_citations = result["citations"]
                answer_text = result["answer"]

        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})

# ── RIGHT: Evidence panel ──────────────────────────────────────────────────────
with evidence_col:
    st.markdown("### 📌 Source Evidence")

    citations = st.session_state.last_citations
    if not citations:
        st.info("Ask a question to see the source passages used to generate the answer.")
    else:
        st.caption(f"{len(citations)} passage(s) retrieved:")
        for c in citations:
            with st.expander(f"[{c['index']}] {c['label']}"):
                snippet = c["text"].strip()
                if len(snippet) > 700:
                    snippet = snippet[:700] + "…"
                st.markdown(snippet)
