"""
ingest.py
---------
Handles loading course material (txt or pdf), chunking it into coherent
sections, embedding those chunks, and storing them in a local Chroma
vector database.

Run directly to (re)build the index from everything in the data/ folder:

    python src/ingest.py
"""

import os
import re
import glob
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "course_material"

# Free, local embedding model - no API key required for this step.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents() -> list[dict]:
    """Loads every .txt and .pdf file in data/ and returns
    a list of {source, text} dicts."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*")):
        if path.endswith(".txt"):
            docs.append({"source": os.path.basename(path), "text": read_txt(path)})
        elif path.endswith(".pdf"):
            docs.append({"source": os.path.basename(path), "text": read_pdf(path)})
    return docs


def chunk_by_section(text: str, source: str) -> list[dict]:
    """
    Splits text into chunks along 'Section' headers when present, otherwise
    falls back to paragraph-based chunking. Keeping chunks concept-aligned
    matters a lot for a Socratic tutor, since each hint should map to one
    coherent idea rather than a fragment split across two chunks.
    """
    section_pattern = re.compile(r"(?=Section \d+:)")
    parts = section_pattern.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # Fallback: split on blank lines (paragraphs)
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for i, part in enumerate(parts):
        title_match = re.match(r"Section \d+: (.+)", part)
        title = title_match.group(1) if title_match else f"Chunk {i+1}"
        chunks.append({
            "id": f"{source}::{i}",
            "text": part,
            "source": source,
            "title": title,
        })
    return chunks


def build_index(reset: bool = True) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR}. Add .txt or .pdf files and rerun.")
        return collection

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_by_section(doc["text"], doc["source"]))

    if all_chunks:
        collection.add(
            ids=[c["id"] for c in all_chunks],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{"source": c["source"], "title": c["title"]} for c in all_chunks],
        )
        print(f"Indexed {len(all_chunks)} chunks from {len(docs)} document(s).")

    return collection


if __name__ == "__main__":
    build_index(reset=True)
