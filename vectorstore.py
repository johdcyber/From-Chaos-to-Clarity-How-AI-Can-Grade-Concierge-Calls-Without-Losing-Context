from typing import List, Dict, Any, Optional
import numpy as np

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

class InMemoryVectorStore:
    """A minimal vector store for demo/testing. For production, use Pinecone/FAISS/etc."""
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = None  # np.ndarray [N, dim]
        self._meta: List[Dict[str, Any]] = []

    def upsert(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        if self._vecs is None:
            self._vecs = vectors.copy()
        else:
            self._vecs = np.vstack([self._vecs, vectors])
        self._meta.extend(metadatas)

    def query(self, vector: np.ndarray, top_k: int = 5, filter: Optional[Dict[str, Any]] = None):
        if self._vecs is None or len(self._meta) == 0:
            return []
        sims = cosine_sim(vector.reshape(1, -1), self._vecs).flatten()
        idxs = np.argsort(-sims)
        results = []
        for idx in idxs:
            meta = self._meta[idx]
            if filter:
                ok = True
                for k, v in filter.items():
                    if meta.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            results.append({"score": float(sims[idx]), "metadata": meta, "index": int(idx)})
            if len(results) >= top_k:
                break
        return results

# Pinecone stub (for real integration, uncomment and implement):
#
# import pinecone
# class PineconeVectorStore:
#     def __init__(self, index_name: str, dim: int, api_key: str, environment: str):
#         pinecone.init(api_key=api_key, environment=environment)
#         self.index = pinecone.Index(index_name)
#         self.dim = dim
#     def upsert(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
#         items = []
#         for i, (v, m) in enumerate(zip(vectors, metadatas)):
#             items.append((m["id"], v.tolist(), m))
#         self.index.upsert(vectors=items)
#     def query(self, vector: np.ndarray, top_k: int = 5, filter: Optional[Dict[str, Any]] = None):
#         res = self.index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True, filter=filter or {})
#         return [{"score": m.score, "metadata": m.metadata} for m in res.matches]