"""Knowledge base ingestion for RAG."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from rag.vector_store import FaissVectorStore


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Chunk text into overlapping windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_docs(knowledge_dir: str = "knowledge_base") -> List[Dict[str, Any]]:
    """Load markdown docs and generate chunk records."""
    records: List[Dict[str, Any]] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(raw)):
            records.append({
                "content": chunk,
                "source": str(path),
                "chunk_id": i,
                "title": path.stem,
            })
    return records


def ingest_knowledge_base(knowledge_dir: str = "knowledge_base") -> None:
    """Ingest KB docs into FAISS index."""
    records = load_docs(knowledge_dir)
    store = FaissVectorStore()
    store.build(records)
    store.save()


if __name__ == "__main__":
    ingest_knowledge_base("knowledge_base")
    print("Knowledge base ingested into FAISS.")
