from typing import List
import numpy as np
import hashlib

class LocalStubEmbeddings:
    """A tiny, deterministic embedding stub (hash-based) to keep the demo runnable offline.
    Replace with SentenceTransformers or OpenAI embeddings in production."""
    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            # Produce a 256-dim vector by hashing tokens; purely deterministic stub.
            h = hashlib.sha256(t.encode("utf-8")).digest()
            base = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            # Repeat to reach 256 dims
            v = np.tile(base, 8)[:256]
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.vstack(vecs)

# Example for real embedding usage (commented for offline demo):
#
# from sentence_transformers import SentenceTransformer
# class SentenceTransformersEmbeddings:
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#     def embed(self, texts: List[str]) -> np.ndarray:
#         return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
#
# import openai
# class OpenAIEmbeddings:
#     def __init__(self, model: str = "text-embedding-3-large"):
#         self.model = model
#     def embed(self, texts: List[str]) -> np.ndarray:
#         # Call OpenAI embeddings API and return np.ndarray
#         raise NotImplementedError("Integrate with your API client here.")