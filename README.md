# BioQuery 🧬📄  
## Biology Research Paper Assistant (RAG with Citations)

**BioQuery** is a document-grounded question-answering assistant for **biology research papers**.  
It allows users to upload one or multiple PDFs, build a local FAISS index, and ask natural-language questions.  
The system responds with **evidence-backed answers and page-level citations**, ensuring reliability and transparency.

This project is built as a **portfolio-grade demonstration of Retrieval-Augmented Generation (RAG)**.

---

## ✨ Key Features

- **Multi-PDF Paper Library**
  - Upload and query multiple research papers simultaneously

- **Citation-First Answers**
  - Each answer includes:
    - PDF name
    - Page number
    - Evidence snippet used to generate the response

- **Evidence-Grounded Reasoning**
  - Answers are generated **only from retrieved document chunks**

- **“I Don’t Know” Safety**
  - If the answer is not supported by the selected paper(s), the assistant refuses to hallucinate

- **Fast Local Retrieval**
  - FAISS vector index stored locally for low-latency semantic search

---

## 🧱 Tech Stack

- **Python 3.11+**
- **Streamlit** — interactive web UI
- **FAISS** — vector database for similarity search
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — embeddings
- **Groq API** — fast LLM inference
- **LangChain** — retrieval and orchestration utilities

---

## 📂 Project Structure

RAG_BIOO/
├── app/
│ ├── ingest.py # Build FAISS index from PDFs
│ ├── query.py # CLI-based Q&A
│ ├── ui.py # Streamlit UI
│ └── utils.py # Shared helper functions
│
├── data/
│ └── raw_pdfs/ # Uploaded research papers
│
├── vectorstore/ # Saved FAISS index (not committed)
│
├── .env # API keys (not committed)
├── requirements.txt
└── README.md

---

## 🚀 Setup & Run (Local)

### 1️⃣ Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

### 2️⃣ Install dependencies
python -m pip install -r requirements.txt

### 3️⃣ Add Groq API key
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here

### 4️⃣ Add research PDFs
Place one or more PDFs inside:
data/raw_pdfs/
Example:
cp "/path/to/Ref 1.pdf" data/raw_pdfs/
cp "/path/to/Ref 2.pdf" data/raw_pdfs/

### 5️⃣ Build FAISS index
python app/ingest.py

### 6️⃣ Run the application
python -m streamlit run app/ui.py

### What This Project Demonstrates
End-to-end RAG pipeline (PDF → embeddings → FAISS → LLM)
Vector database usage with FAISS
Citation-backed and evidence-grounded responses
Safe AI behavior (hallucination prevention)
Production-quality practices:
virtual environments
environment variable management
clean repository structure
reproducible setup





