"""RAG retrieval helpers."""

from __future__ import annotations

from typing import List

from rag.vector_store import FaissVectorStore, RetrievedChunk


class RAGRetriever:
    """Top-k retriever over local FAISS index."""

    def __init__(self):
        self.store = FaissVectorStore()
        self.ready = self.store.load()

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if not self.ready:
            return []
        return self.store.search(query=query, top_k=top_k)
