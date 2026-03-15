"""
query.py — Retrieval and answer generation for BioQuery

This module handles two main tasks:
1. Retrieving the most relevant text chunks from the FAISS index for a question.
2. Sending those chunks + the question to the Groq LLM and returning the answer.

Important: All logic is inside functions — nothing runs at import time.
This prevents crashes when the FAISS index hasn't been built yet.
"""

import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_vectorstore_dir, format_source_label


# ---------------------------------------------------------------------------
# System prompts — these tell the LLM how to behave
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """You are BioQuery, a friendly and knowledgeable biology research assistant.

Your job is to answer questions based ONLY on the provided context from uploaded biology research papers.

Follow these rules strictly:
1. Use ONLY the information given in the context. Do not use outside knowledge.
2. If the answer is not clearly present in the context, say exactly:
   "I don't know based on the provided paper(s). This information was not found in the uploaded document."
3. Write in simple, clear English that a biology student can understand.
4. Always cite your sources using the chunk numbers like [1], [2] at the end of sentences.
5. Structure your answer clearly:
   - Start with a direct 2-3 sentence answer.
   - Then list key points as bullets if there are multiple points.
   - End with the source references used.
6. Be honest. Never guess or make things up.
"""

SUMMARY_SYSTEM_PROMPT = """You are BioQuery, a biology research assistant helping students understand research papers.

Your job is to write a clear, student-friendly summary of the provided content from a biology research paper.

Structure your summary like this:
📌 **What this paper is about** (2-3 sentences)
🎯 **Main objective / research question**
🔬 **Methods used** (how they did the research)
📊 **Key findings / results**
✅ **Conclusions**
📄 **Source references** (mention page numbers where information came from)

Write in simple English. Avoid jargon. A first-year biology student should be able to understand this summary.
"""


# ---------------------------------------------------------------------------
# Helper: Load the embedding model (cached in memory after first call)
# ---------------------------------------------------------------------------

_embeddings = None  # module-level cache so we don't reload the model every time

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads and returns the Sentence-Transformer embedding model.
    The model is loaded only once and reused, which is much faster.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Vectorstore: Load from disk
# ---------------------------------------------------------------------------

def load_vectorstore() -> FAISS:
    """
    Loads the FAISS index from disk and returns it.

    This is called inside a function (not at module level) so the app doesn't
    crash when the index hasn't been built yet.
    """
    vectorstore_dir = str(get_vectorstore_dir())
    embeddings = get_embeddings()
    return FAISS.load_local(
        vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # safe because we built this index ourselves
    )


# ---------------------------------------------------------------------------
# Retrieval: Get the most relevant text chunks for a question
# ---------------------------------------------------------------------------

def retrieve_chunks(db: FAISS, query: str, k: int = 4) -> list:
    """
    Retrieves the top-k most relevant text chunks for a given question.

    Uses similarity_search_with_score so we get real distance scores
    (lower score = more similar for FAISS L2 distance).

    Returns a list of (Document, score) tuples.
    """
    # similarity_search_with_score returns (document, distance_score)
    docs_and_scores = db.similarity_search_with_score(query, k=k)
    return docs_and_scores


def get_confidence_level(scores: list[float]) -> tuple[str, str]:
    """
    Maps FAISS distance scores to a human-readable confidence level.

    FAISS uses L2 distance: lower score = better match.
    These thresholds work well for the MiniLM-L6-v2 embedding model.

    Returns: (label, emoji) e.g. ("High", "🟢")
    """
    if not scores:
        return "Low", "🔴"

    best_score = scores[0]

    if best_score < 0.5:
        return "High", "🟢"
    elif best_score < 1.0:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"


# ---------------------------------------------------------------------------
# Context formatting: Turn retrieved chunks into a readable string for the LLM
# ---------------------------------------------------------------------------

def build_context_string(docs_and_scores: list) -> tuple[str, list]:
    """
    Converts retrieved (Document, score) pairs into:
    1. A formatted context string to send to the LLM.
    2. A list of citation tuples for display in the UI.

    Each citation tuple: (index, source_label, page_text_snippet, score)
    """
    context_parts = []
    citations = []

    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        source_label = format_source_label(doc.metadata)
        text = doc.page_content.strip().replace("\n", " ")

        # Numbered context block for the LLM
        context_parts.append(f"[{i}] Source: {source_label}\n{text}")

        # Citation info for the UI sidebar
        citations.append({
            "index": i,
            "label": source_label,
            "text": doc.page_content.strip(),
            "score": score,
        })

    return "\n\n".join(context_parts), citations


# ---------------------------------------------------------------------------
# LLM calls: Ask Groq for an answer or summary
# ---------------------------------------------------------------------------

def call_groq(system_prompt: str, user_message: str, model: str) -> str:
    """
    Calls the Groq LLM API and returns the response text.

    Args:
        system_prompt: Instructions telling the LLM how to behave.
        user_message: The actual question/context to respond to.
        model: The Groq model name to use.

    Returns:
        The LLM's response as a plain string.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

    client = Groq(api_key=groq_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,  # temperature=0 keeps answers consistent and factual
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def answer_question(db: FAISS, question: str, k: int = 4, model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Full Q&A pipeline: retrieve → format context → generate answer.

    Args:
        db: Loaded FAISS vectorstore.
        question: The user's question.
        k: Number of chunks to retrieve.
        model: Groq model to use for generation.

    Returns:
        A dict with: answer, citations, confidence_label, confidence_emoji
    """
    # Step 1: Retrieve relevant chunks
    docs_and_scores = retrieve_chunks(db, question, k=k)

    # Step 2: Format the context string and citation data
    context_string, citations = build_context_string(docs_and_scores)

    # Step 3: Calculate confidence from retrieval scores
    scores = [s for _, s in docs_and_scores]
    confidence_label, confidence_emoji = get_confidence_level(scores)

    # Step 4: Build the user message for the LLM
    user_message = (
        f"Context from the uploaded paper(s):\n\n"
        f"{context_string}\n\n"
        f"Question: {question}\n\n"
        f"Please answer using the format described. Cite sources as [1], [2], etc."
    )

    # Step 5: Get the answer from Groq
    answer = call_groq(QA_SYSTEM_PROMPT, user_message, model)

    return {
        "answer": answer,
        "citations": citations,
        "confidence_label": confidence_label,
        "confidence_emoji": confidence_emoji,
    }


def summarize_paper(db: FAISS, model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Generates a structured summary of the uploaded paper(s).

    Strategy: We retrieve a broad set of chunks using generic terms that
    typically appear across a biology paper (abstract, methods, results, conclusions).
    This gives the LLM a wide view of the paper to summarize from.

    Returns:
        A dict with: summary, citations
    """
    # Use broad summary-related queries to get chunks from different paper sections
    summary_queries = [
        "introduction background objective of the study",
        "methods materials experimental design dataset",
        "results findings observations data analysis",
        "conclusion discussion future work",
    ]

    seen_texts = set()
    all_docs_and_scores = []

    for query in summary_queries:
        docs_and_scores = retrieve_chunks(db, query, k=3)
        for doc, score in docs_and_scores:
            # Avoid duplicate chunks (same text from multiple queries)
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                all_docs_and_scores.append((doc, score))

    if not all_docs_and_scores:
        return {"summary": "Could not retrieve enough content to generate a summary.", "citations": []}

    context_string, citations = build_context_string(all_docs_and_scores)

    user_message = (
        f"Here are excerpts from a biology research paper:\n\n"
        f"{context_string}\n\n"
        f"Please write a complete, student-friendly summary of this paper using the format provided."
    )

    summary = call_groq(SUMMARY_SYSTEM_PROMPT, user_message, model)

    return {"summary": summary, "citations": citations}
