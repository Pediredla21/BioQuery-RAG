# BioQuery 🧬📄 — Biology Paper RAG Assistant (with Citations)

BioQuery is a **document-grounded question-answering assistant** for biology research papers.
You can upload one or multiple PDFs, build a local FAISS index, and ask questions — the app responds with:
- ✅ Answers based only on the selected papers
- ✅ Page-level citations
- ✅ Evidence snippets (retrieved text used to generate the answer)

This project is built as a portfolio-grade demo of **RAG (Retrieval-Augmented Generation)** using:
- **FAISS** for vector search
- **Sentence-Transformers embeddings** for semantic retrieval
- **Groq LLM** for fast generation
- **Streamlit** for an interactive UI

---

## ✨ Key Features

- **Multi-PDF Paper Library**
  - Upload multiple papers
  - Select which papers to index and search
- **Citation-First Answers**
  - Shows sources as: `PDF name + Page number`
  - Displays evidence snippets used for answering
- **“I don’t know” Safety**
  - If the answer is not supported by the selected paper(s), the assistant refuses or marks low confidence
- **Fast Local Retrieval**
  - FAISS index stored locally in `vectorstore/`

---

## 🧱 Tech Stack

- Python 3.11+
- Streamlit (UI)
- FAISS (vector database)
- Sentence-Transformers (`all-MiniLM-L6-v2`) for embeddings
- Groq API for LLM responses

---

## 📂 Project Structure

RAG_BIOO/
app/
ingest.py # Build FAISS index from PDFs
query.py # CLI Q&A (loads FAISS + asks Groq)
ui.py # Streamlit UI
utils.py
data/
raw_pdfs/ # Uploaded PDFs stored here
vectorstore/ # Saved FAISS index
.env # API keys (not committed)
requirements.txt
README.md


---

## 🚀 Setup & Run (Local)

### 1) Create & activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

2) Install dependencies
python -m pip install -r requirements.txt

3) Add Groq API key

Create .env:

GROQ_API_KEY=your_key_here

4) Put PDFs into data/raw_pdfs/

Example:

cp "/path/to/Ref 1.pdf" data/raw_pdfs/
cp "/path/to/Ref 2.pdf" data/raw_pdfs/

5) Build index
python app/ingest.py

6) Run UI
python -m streamlit run app/ui.py


Open:

http://localhost:8501

🧪 Demo (Screenshot / Video)
Add a screenshot

Create a folder:

assets/


Save an image:

assets/demo.png


Then embed in README:

![BioQuery Demo](assets/demo.png)

Add a short GIF (recommended)

Record a quick screen capture (10–20 sec) and convert to GIF as assets/demo.gif, then:

![BioQuery GIF](assets/demo.gif)

✅ What This Project Demonstrates (for Recruiters)

End-to-end RAG pipeline

Vector search with FAISS

Strong UX: upload → index → ask → cite evidence

Safe behavior: refuse when evidence is missing

Production habits: environment variables, clean folder structure, reproducible setup

🔮 Future Improvements

Compare mode: “Ref1 vs Ref2 answer comparison”

Export chat to Markdown/PDF

Per-document FAISS indexes + caching

Add evaluation: retrieval accuracy & citation correctness


---

## 6) Right now: “I want upload these two and ask questions”

If your UI already looks like your last screenshot, do this flow:

1) Upload both PDFs  
2) Click