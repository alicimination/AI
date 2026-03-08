"""Vector store abstraction for FAISS + sentence-transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    content: str
    metadata: Dict[str, Any]
    score: float


class FaissVectorStore:
    """Simple FAISS-backed vector store."""

    def __init__(self, index_path: str = "rag/faiss.index", meta_path: str = "rag/faiss_meta.pkl"):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.metadatas: List[Dict[str, Any]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vectors, dtype="float32")

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        contents = [c["content"] for c in chunks]
        vectors = self._embed(contents)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadatas = chunks

    def save(self) -> None:
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self) -> bool:
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        with self.meta_path.open("rb") as f:
            self.metadatas = pickle.load(f)
        return True

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.index is None or not self.metadatas:
            return []
        q = self._embed([query])
        scores, indices = self.index.search(q, top_k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            results.append(RetrievedChunk(content=meta["content"], metadata=meta, score=float(score)))
        return results
