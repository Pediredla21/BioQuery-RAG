import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_DIR = "data/raw_pdfs"
VECTORSTORE_DIR = "vectorstore"

def ingest_pdfs():
    documents = []

    # 1) Load PDFs (page by page)
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            documents.extend(pages)

    if not documents:
        raise FileNotFoundError(
            f"No PDFs found in '{PDF_DIR}'. Add at least one PDF and try again."
        )

    print(f"Loaded {len(documents)} pages")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")

    # 3) Local embeddings (FREE) + stable CPU config
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # 4) Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5) Save FAISS index
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print("Vector store saved successfully ✅")
    print(f"Saved to: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    ingest_pdfs()
