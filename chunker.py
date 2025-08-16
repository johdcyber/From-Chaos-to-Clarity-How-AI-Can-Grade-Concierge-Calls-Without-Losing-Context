from typing import List, Dict, Tuple

def simple_tokenize(text: str) -> List[str]:
    """A naive tokenizer (space split) used for the runnable demo.
    Swap with `tiktoken` or your preferred tokenizer for production."""
    return text.split()

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

def chunk_by_tokens(text: str, chunk_size: int = 120, overlap: int = 30) -> List[Dict]:
    """Split `text` into chunks by token count with overlap.
    Returns a list of dicts: {"chunk_id": int, "start": int, "end": int, "text": str}
    """
    tokens = simple_tokenize(text)
    n = len(tokens)
    chunks = []
    start = 0
    cid = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        chunks.append({
            "chunk_id": cid,
            "start": start,
            "end": end,
            "text": detokenize(chunk_tokens),
        })
        cid += 1
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def neighbor_window(chunks: List[Dict], i: int, window: int = 1) -> Tuple[int, int]:
    """Return (lo, hi) inclusive indices for ±window neighbor expansion."""
    lo = max(0, i - window)
    hi = min(len(chunks) - 1, i + window)
    return lo, hi

def expand_with_neighbors(chunks: List[Dict], i: int, window: int = 1) -> str:
    """Concatenate the target chunk with its neighbors for grading context."""
    lo, hi = neighbor_window(chunks, i, window)
    texts = [chunks[j]["text"] for j in range(lo, hi + 1)]
    return "\n".join(texts)