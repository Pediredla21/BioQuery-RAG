import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PDF_DIR = Path("data/raw_pdfs")
VECTORSTORE_DIR = Path("vectorstore")

st.set_page_config(page_title="BioQuery • Research RAG", layout="wide")

# ---------- Core (cached) ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

def vectorstore_exists() -> bool:
    return (VECTORSTORE_DIR / "index.faiss").exists() and (VECTORSTORE_DIR / "index.pkl").exists()

def list_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.name for p in PDF_DIR.glob("*.pdf")])

def ingest_selected_pdfs(selected_files):
    if not selected_files:
        return False, "Please select at least one PDF."

    docs = []
    for fname in selected_files:
        loader = PyPDFLoader(str(PDF_DIR / fname))
        docs.extend(loader.load())

    if not docs:
        return False, "No pages were loaded. Check your PDF file(s)."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTORSTORE_DIR))

    return True, f"Indexed {len(selected_files)} PDF(s) • {len(docs)} pages • {len(chunks)} chunks."

@st.cache_resource
def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True)

def build_context(docs):
    blocks = []
    citations = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "unknown")
        text = d.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] Source: {src} | Page: {page}\n{text}")
        citations.append((i, src, page, d.page_content.strip()))
    return "\n\n".join(blocks), citations

def compress_context_for_question(citations):
    # Keep only strongest ~3 evidence snippets to tightly ground the model
    short = []
    for idx, src, page, raw_text in citations[:3]:
        snippet = raw_text.strip().replace("\n", " ")
        if len(snippet) > 380:
            snippet = snippet[:380] + "…"
        short.append(f"[{idx}] ({src} p{page}) {snippet}")
    return "\n".join(short)

def confidence_label(scores):
    """
    In FAISS similarity_search_with_score, the score is often distance-like:
    lower score = closer. We'll keep heuristic thresholds.
    If your scores behave opposite, adjust thresholds.
    """
    if not scores:
        return "Low", "🔴"

    best = scores[0]
    if best < 0.35:
        return "High", "🟢"
    if best < 0.55:
        return "Medium", "🟡"
    return "Low", "🔴"

def groq_answer(question, full_context, evidence_snippets, model_name="llama-3.3-70b-versatile"):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY missing. Add it to .env")

    client = Groq(api_key=key)

    system = (
        "You are BioQuery, a careful biology research assistant.\n"
        "You MUST follow these rules:\n"
        "1) Use ONLY the provided context from the selected paper(s).\n"
        "2) If the answer is not clearly supported in the context, say exactly:\n"
        "   \"I don't know based on the provided paper(s).\"\n"
        "3) Write in simple, student-friendly English.\n"
        "4) Always include citations like [1], [2] next to the sentence they support.\n"
        "5) Output MUST follow this exact format:\n"
        "   Direct Answer: (2–4 lines)\n"
        "   Key Points:\n"
        "   - bullet 1\n"
        "   - bullet 2\n"
        "   Evidence:\n"
        "   - \"short quote/snippet\" [#]\n"
        "   - \"short quote/snippet\" [#]\n"
    )

    user = (
        f"Question: {question}\n\n"
        f"Top Evidence Snippets (most important):\n{evidence_snippets}\n\n"
        f"Full Context:\n{full_context}\n\n"
        "Now answer using the required format."
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    t1 = time.time()
    return resp.choices[0].message.content, (t1 - t0)

# ---------- UI ----------
st.markdown(
    """
    <div style="padding: 10px 0;">
      <h1 style="margin-bottom: 0;">🧬 BioQuery</h1>
      <p style="margin-top: 6px; color: #9aa4b2;">
        Document-grounded RAG assistant • Better answers • Citations + evidence snippets • Safer (anti-hallucination)
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📄 Paper Library")

    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Save Uploads"):
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in uploaded or []:
            out = PDF_DIR / f.name
            out.write_bytes(f.read())
            count += 1
        st.success(f"Saved {count} PDF(s) into {PDF_DIR}")
        st.rerun()

    pdfs = list_pdfs()
    if not pdfs:
        st.info("No PDFs found yet. Upload PDFs or place them in `data/raw_pdfs/`.")

    selected = st.multiselect("Select PDFs to index/search", options=pdfs, default=pdfs[:2] if len(pdfs) >= 2 else pdfs)

    st.divider()
    st.subheader("🧱 Index")
    st.caption("Build FAISS vector index for selected PDFs.")
    if st.button("Build / Rebuild Index"):
        ok, msg = ingest_selected_pdfs(selected)
        if ok:
            st.success(msg)
            load_vectorstore.clear()
        else:
            st.error(msg)

    st.write("**Index status:**", "✅ Ready" if vectorstore_exists() else "⚠️ Not built yet")

    st.divider()
    st.subheader("⚙️ Settings")
    model = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], index=0)
    k = st.slider("Top-k passages", 2, 8, 4)
    strict_mode = st.toggle("Strict grounded mode (refuse if evidence weak)", value=True)

# chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

if not vectorstore_exists():
    st.warning("Build the index first using the sidebar → **Build / Rebuild Index**.")
    st.stop()

chat_col, sources_col = st.columns([1.15, 0.85])

with chat_col:
    st.subheader("💬 Chat with your paper(s)")
    st.caption("Tip: Ask questions directly related to the selected papers (methods, datasets, results, conclusions).")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask a question (e.g., 'Summarize antimicrobial peptides discussed in the paper')")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        db = load_vectorstore()

        # ---- Retrieval (MMR for better evidence diversity) ----
        t0 = time.time()
        try:
            docs = db.max_marginal_relevance_search(user_q, k=k, fetch_k=20)
            docs_and_scores = [(d, 0.45) for d in docs]  # dummy scores since MMR doesn't return scores
        except Exception:
            # fallback if MMR not available
            docs_and_scores = db.similarity_search_with_score(user_q, k=k)
            docs = [d for d, _ in docs_and_scores]

        t1 = time.time()

        full_context, citations = build_context(docs)
        evidence_snippets = compress_context_for_question(citations)

        # ---- Confidence (use heuristics; if dummy, treat as Medium) ----
        scores = [s for _, s in docs_and_scores] if docs_and_scores else []
        conf, badge = confidence_label(scores) if scores else ("Medium", "🟡")

        # ---- Grounded refusal rule ----
        refuse = strict_mode and (conf == "Low")

        if refuse:
            assistant_text = (
                "I don't know based on the provided paper(s).\n\n"
                "_Reason_: The selected paper(s) do not contain enough focused evidence to answer reliably."
            )
            llm_time = 0.0
        else:
            assistant_text, llm_time = groq_answer(
                question=user_q,
                full_context=full_context,
                evidence_snippets=evidence_snippets,
                model_name=model,
            )

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        with st.chat_message("assistant"):
            st.markdown(f"**Confidence:** {badge} **{conf}**")
            st.markdown(assistant_text)

            # show sources summary inline
            unique_sources = sorted({f"{src} p{page}" for _, src, page, _ in citations})
            if unique_sources:
                st.markdown("**Sources used:** " + ", ".join(unique_sources))

            st.caption(f"Retrieval: {t1-t0:.2f}s • LLM: {llm_time:.2f}s • Top-k: {k}")

        st.session_state.last_citations = citations

with sources_col:
    st.subheader("📌 Evidence & Citations")

    citations = st.session_state.get("last_citations", [])
    if not citations:
        st.info("Ask a question to see supporting evidence here.")
    else:
        for idx, src, page, raw_text in citations:
            with st.expander(f"[{idx}] {src} • Page {page}"):
                snippet = raw_text.strip()
                if len(snippet) > 1100:
                    snippet = snippet[:1100] + "…"
                st.write(snippet)
