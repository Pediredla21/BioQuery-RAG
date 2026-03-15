"""
ingest.py — PDF loading, chunking, and FAISS index building for BioQuery

How it works (simple explanation):
1. Load each PDF page-by-page using PyPDFLoader.
2. Enrich each page with clean metadata (source file name, page number).
3. Split pages into smaller text chunks for better retrieval.
4. Convert chunks to vector embeddings using a Sentence-Transformer model.
5. Store all embeddings in a FAISS index and save it to disk.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import shared path helpers from utils
from utils import get_pdf_dir, get_vectorstore_dir


def load_pdfs(pdf_paths: list[str]) -> list:
    """
    Loads a list of PDF file paths and returns all pages as LangChain Documents.

    Each document (page) automatically gets metadata like:
    - source: full file path
    - page: page index (0-based from PyPDFLoader)

    We also add 'source_name' (just the filename) for cleaner display in the UI.
    """
    all_pages = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Add a clean 'source_name' field (just the filename, not the full path)
        filename = os.path.basename(pdf_path)
        for page in pages:
            page.metadata["source_name"] = filename

        all_pages.extend(pages)
        print(f"  Loaded {len(pages)} pages from '{filename}'")

    return all_pages


def split_into_chunks(documents: list) -> list:
    """
    Splits the loaded PDF pages into smaller text chunks.

    Why chunking?
    A full PDF is too large to fit in one embedding. Smaller chunks allow the
    retrieval system to find the most relevant passage for a given question.

    chunk_size=800: each chunk is ~800 characters (~100-130 words)
    chunk_overlap=150: chunks overlap slightly so context isn't lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} text chunks from {len(documents)} pages")
    return chunks


def build_faiss_index(chunks: list, vectorstore_dir: Path) -> None:
    """
    Converts text chunks to embeddings and builds a FAISS vector index.

    The embedding model (all-MiniLM-L6-v2) converts each text chunk into a
    384-dimensional vector. FAISS then allows fast similarity search over these vectors.
    """
    print("  Loading embedding model (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print("  Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index to disk so the UI can load it without re-indexing every time
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    print(f"  FAISS index saved to: {vectorstore_dir}")


def ingest_pdfs_from_paths(pdf_paths: list[str]) -> tuple[bool, str]:
    """
    Main ingestion function called by the Streamlit UI.

    Takes a list of absolute PDF file paths, processes them, and builds the FAISS index.

    Returns:
        (True, success_message) on success
        (False, error_message) on failure
    """
    if not pdf_paths:
        return False, "No PDFs provided. Please upload at least one PDF."

    print(f"\n=== Starting ingestion for {len(pdf_paths)} PDF(s) ===")

    # Step 1: Load all PDF pages
    documents = load_pdfs(pdf_paths)

    if not documents:
        return False, "No pages were loaded. The PDFs may be empty or unreadable."

    # Step 2: Split pages into chunks
    chunks = split_into_chunks(documents)

    # Step 3: Build and save the FAISS index
    vectorstore_dir = get_vectorstore_dir()
    build_faiss_index(chunks, vectorstore_dir)

    msg = (
        f"Successfully indexed {len(pdf_paths)} PDF(s) → "
        f"{len(documents)} pages → {len(chunks)} chunks."
    )
    print(f"\n✅ {msg}")
    return True, msg


def ingest_all_pdfs_in_dir() -> tuple[bool, str]:
    """
    Convenience function for CLI use: ingests all PDFs in the data/raw_pdfs/ folder.
    """
    pdf_dir = get_pdf_dir()
    pdf_paths = [str(p) for p in pdf_dir.glob("*.pdf")]

    if not pdf_paths:
        return False, f"No PDFs found in '{pdf_dir}'. Add at least one PDF."

    return ingest_pdfs_from_paths(pdf_paths)


# Allow running this file directly from the command line: python app/ingest.py
if __name__ == "__main__":
    ok, message = ingest_all_pdfs_in_dir()
    print(message)
