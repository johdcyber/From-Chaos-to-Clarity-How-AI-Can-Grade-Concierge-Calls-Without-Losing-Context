
# Call QA – Map → Reduce → Refine (Developer Guide + Reference Implementation)

This repository is a **developer how‑to** for grading long call transcripts without losing context.
It implements a minimal, runnable version of the **Map → Reduce → Refine** pipeline using:

- Token‑aware chunking with overlap + neighbor expansion
- Embedding + vector store retrieval (in‑memory demo; Pinecone stub included)
- Hybrid scoring (deterministic regex checks + LLM‑like stub)
- Call‑level aggregation and a final *Refine* stage with a balanced scorecard

> You can swap the stubs with your preferred ASR, embedding model, LLM, and vector database (e.g., Pinecone).

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Output**: Console JSON report for the call, including chunk scores, call-level rules, and final QA verdict.

## Files

- `chunker.py` — tokenization and chunking with overlap and neighbor expansion
- `embeddings.py` — pluggable embeddings (local stub + SentenceTransformers example + OpenAI example)
- `vectorstore.py` — in‑memory vector store (runnable) + Pinecone stub (drop‑in)
- `scoring.py` — deterministic checks + chunk grading stub
- `map_reduce_refine.py` — orchestrates Map → Reduce → Refine
- `prompts.py` — example prompts and rubrics (for real LLMs)
- `main.py` — end‑to‑end demo using `data/sample_transcript.txt`
- `requirements.txt` — optional libraries if you want to replace stubs
- `data/sample_transcript.txt` — synthetic call with a late disclosure

## Swap-in Guidance

- **ASR**: Replace `data/sample_transcript.txt` with your transcript (or generate from audio using your ASR of choice).
- **Embeddings**: Replace `LocalStubEmbeddings` with `SentenceTransformersEmbeddings` or OpenAI embeddings.
- **Vector DB**: Replace `InMemoryVectorStore` with `PineconeVectorStore` (see stub).
- **LLM**: Replace the stub in `scoring.grade_chunk_llm_like` with a real LLM call using `prompts.CHUNK_RUBRIC`.

## Notes
- The demo uses simple tokenization to stay runnable without external dependencies.
- This project is for educational purposes. Add your own auth, privacy filtering, and evaluation before production.
