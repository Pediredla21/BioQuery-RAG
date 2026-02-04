import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_DIR = "vectorstore"

SYSTEM_PROMPT = """You are BioQuery, a research assistant for biology papers.
Answer ONLY using the provided context from the uploaded paper.
If the answer is not in the context, say: "I don't know based on the provided paper."
Keep the answer clear, concise, and factual.
"""

# ✅ Load embeddings ONCE (not every question)
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# ✅ Load FAISS index ONCE
DB = FAISS.load_local(
    VECTORSTORE_DIR,
    EMBEDDINGS,
    allow_dangerous_deserialization=True,
)

def format_context(docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "unknown")
        text = d.page_content.strip().replace("\n", " ")
        parts.append(f"[{i}] Source: {src} | Page: {page}\n{text}\n")
    return "\n".join(parts)

def format_sources(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "unknown")
        lines.append(f"[{i}] {src} (page {page})")
    return "\n".join(lines)

def ask(query: str, k: int = 4):
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    client = Groq(api_key=groq_key)

    # Retrieve top-k chunks
    docs = DB.similarity_search(query, k=k)
    context = format_context(docs)

    user_prompt = f"""Context from paper:
{context}

Question: {query}

Answer with citations by referencing chunk numbers like [1], [2] where relevant.
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content

    print("\n================= ANSWER =================\n")
    print(answer)

    print("\n================ SOURCES ================\n")
    print(format_sources(docs))
    print("\n=========================================\n")

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        ask(q)
