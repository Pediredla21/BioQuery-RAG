"""
utils.py — Shared helpers for BioQuery

This file provides a single place for:
- Getting correct absolute paths (so the app works regardless of where you run it from)
- Formatting source/citation labels for display
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root (the RAG_BIOO folder).
    This makes all other paths work correctly whether you run the app from
    the project root, the app/ folder, or anywhere else.
    """
    # This file lives in RAG_BIOO/app/, so going one level up gives the project root
    return Path(__file__).resolve().parent.parent


def get_pdf_dir() -> Path:
    """Returns the absolute path to the folder where PDFs are stored."""
    return get_project_root() / "data" / "raw_pdfs"


def get_vectorstore_dir() -> Path:
    """Returns the absolute path to the folder where the FAISS index is saved."""
    return get_project_root() / "vectorstore"


def format_source_label(metadata: dict) -> str:
    """
    Creates a clean, readable source label from a chunk's metadata.

    Example output: "my_paper.pdf • Page 3"

    Args:
        metadata: The metadata dict from a LangChain Document object.

    Returns:
        A formatted string like "filename.pdf • Page N".
    """
    # Use source_name if we added it during ingestion; fall back to raw source path
    source = metadata.get("source_name") or os.path.basename(metadata.get("source", "Unknown"))
    page = metadata.get("page", "?")

    # FAISS page numbers are 0-indexed in PyPDFLoader, so add 1 for human readability
    if isinstance(page, int):
        page = page + 1

    return f"{source} • Page {page}"
