# BioQuery 🧬 — Biology Research Paper RAG Assistant

<div align="center">

**Ask questions about biology research papers and get cited, grounded answers.**

Upload PDFs → Build vector index → Ask questions in natural language → Get answers with page citations

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green?style=flat-square)](https://faiss.ai)
[![Groq](https://img.shields.io/badge/Groq-LLM_API-orange?style=flat-square)](https://console.groq.com)

</div>

---

## 📖 What is BioQuery?

BioQuery is a **RAG (Retrieval-Augmented Generation)** application that lets you chat with biology research papers. Instead of reading a 30-page paper yourself, you can:

- Ask **"Summarize this paper"** → get a structured, student-friendly summary
- Ask **"What methods were used?"** → get a cited answer pointing to the exact page
- Ask **"What are the main findings?"** → get bullet-pointed results with source references
- Ask **anything outside the paper** → get an honest "I don't know based on this paper" response (no hallucination)

The system only answers from the content of your uploaded PDFs — it never makes things up.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-PDF Support** | Upload and index multiple research papers at once |
| **Chat-Driven Summarization** | Just ask "summarize this paper" in the chat — no separate button needed |
| **Page-Level Citations** | Every answer references the source file and page number |
| **Evidence Panel** | See the exact text passages retrieved to generate each answer |
| **Confidence Indicator** | 🟢 High / 🟡 Medium / 🔴 Low confidence based on retrieval similarity |
| **Anti-Hallucination** | Model refuses to answer if evidence isn't found in the paper |
| **Quick-Start Buttons** | Click to pre-fill common questions instantly |
| **Reset Chat** | Clear conversation history with one click |

---

## 🧱 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **UI** | Streamlit | Fast to build, great for ML apps |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Lightweight, runs on CPU, great semantic search |
| **Vector Database** | FAISS | Fast local similarity search, no server needed |
| **PDF Loading** | LangChain + PyPDF | Reliable page-by-page extraction with metadata |
| **LLM** | Groq API (`llama-3.3-70b-versatile`) | Free API, very fast inference |
| **Language** | Python 3.11+ | Standard for ML/AI projects |

---

## 📂 Project Structure

```
BioQuery-RAG/
├── app/
│   ├── utils.py       # Shared helpers: path resolution, citation formatting
│   ├── ingest.py      # PDF loading, chunking, FAISS index building
│   ├── query.py       # Retrieval, answer generation, paper summarization
│   └── ui.py          # Streamlit UI — the main app entry point
│
├── data/
│   └── raw_pdfs/      # Your uploaded PDFs go here (auto-created)
│
├── vectorstore/       # FAISS index files (auto-generated, git-ignored)
│
├── .env               # Your API keys (never committed to git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start — Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Pediredla21/BioQuery-RAG.git
cd BioQuery-RAG
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ **First run note:** The `sentence-transformers` embedding model (~90 MB) will be downloaded automatically on first use. This only happens once.

### 4. Get a free Groq API key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up and create an API key (free)
3. Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 5. Run the app

```bash
streamlit run app/ui.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 🎮 How to Use

| Step | Action |
|---|---|
| 1 | Upload your biology PDF(s) using the sidebar |
| 2 | Click **Save PDFs** |
| 3 | Click **Build / Rebuild Index** |
| 4 | Ask any question in the chat |

**Example questions to try:**
- *"Summarize this paper"*
- *"What is the main research question?"*
- *"What methods or datasets were used?"*
- *"What are the key findings?"*
- *"What are the limitations mentioned?"*
- *"What do the authors conclude?"*

---

## 🧠 How It Works (RAG Architecture)

```
┌─────────────────── INDEXING (done once) ───────────────────┐
│                                                              │
│  PDF File  ──►  Pages  ──►  Chunks (800 chars)             │
│                               │                             │
│                               ▼                             │
│                     Embedding Model                         │
│               (all-MiniLM-L6-v2, 384-dim)                  │
│                               │                             │
│                               ▼                             │
│                    FAISS Vector Index  ──► Saved to disk    │
└──────────────────────────────────────────────────────────────┘

┌─────────────────── QUERYING (every question) ───────────────┐
│                                                              │
│  User Question ──► Embed Question ──► FAISS Search          │
│                                           │                 │
│                                    Top-K Chunks             │
│                              (most similar text passages)   │
│                                           │                 │
│                                  Groq LLM (llama-3.3-70b)  │
│                          Context: chunks + question         │
│                                           │                 │
│                               Grounded Answer + Citations   │
└──────────────────────────────────────────────────────────────┘
```

**Why RAG instead of just asking an LLM?**

A plain LLM (like ChatGPT) answers from its training data and can hallucinate. RAG forces the model to only use the text retrieved from your document, making answers trustworthy and verifiable.

---

## 🔎 Code Walkthrough (for interviews)

### `app/utils.py`
Shared helpers used by all other modules:
- `get_project_root()` — resolves the absolute path to the project root so imports and file paths always work
- `get_pdf_dir()` / `get_vectorstore_dir()` — return correct paths regardless of where you run the app from
- `format_source_label()` — formats a citation label like `"paper.pdf • Page 3"` from LangChain metadata

### `app/ingest.py`
The indexing pipeline (runs when you click "Build Index"):
1. **Load PDFs** using `PyPDFLoader` — extracts text page by page, adds `source_name` to metadata
2. **Chunk** the pages into ~800-character overlapping segments using `RecursiveCharacterTextSplitter`
3. **Embed** each chunk using the `all-MiniLM-L6-v2` model (384-dimensional vectors)
4. **Index** all vectors in FAISS and save to disk

### `app/query.py`
The retrieval and generation pipeline (runs on every question):
1. **Load vectorstore** from disk (only loads once per session)
2. **Embed the question** using the same model as indexing
3. **Retrieve** top-k most similar chunks via FAISS L2 distance search
4. **Compute confidence** from the distance score (lower = better match)
5. **Format context** with numbered citation blocks `[1]`, `[2]`, ...
6. **Call Groq** with a strict system prompt that forces citation-backed answers
7. For summary requests: retrieves from 4 different angles (intro, methods, results, conclusions) to cover the whole paper

### `app/ui.py`
The Streamlit frontend:
- Sidebar: upload + index flow
- Main area: chat interface with quick-start buttons
- Right panel: evidence snippets with expandable source passages
- `is_summary_request()`: keyword detection to route summary vs Q&A


---

## 🔮 Potential Future Improvements

- **Multi-paper comparison** — "How does Paper A's method differ from Paper B?"
- **Chat history export** — download conversation as Markdown or PDF
- **Per-paper indexes** — separate FAISS index per paper for selective searching
- **Evaluation suite** — measure retrieval accuracy and citation correctness
- **Abstract auto-detection** — automatically find and display the paper's abstract

---

## 📄 License

MIT License — free to use, adapt, and build on for your own projects.

---

<div align="center">
Built with ❤️ as a portfolio project for biology research paper analysis.
</div>
