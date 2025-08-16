from typing import List, Dict, Any
from chunker import chunk_by_tokens, expand_with_neighbors
from embeddings import LocalStubEmbeddings
from vectorstore import InMemoryVectorStore
from scoring import deterministic_checks, grade_chunk_llm_like, combine_scores
import numpy as np
import json

def map_phase(call_id: str, transcript: str, chunk_size=120, overlap=30, neighbor_window=1):
    """Chunk transcript, expand with neighbors, grade locally. Returns chunks + embeddings + scores."""
    chunks = chunk_by_tokens(transcript, chunk_size=chunk_size, overlap=overlap)
    expanded_texts = [expand_with_neighbors(chunks, i, window=neighbor_window) for i in range(len(chunks))]

    embedder = LocalStubEmbeddings()
    vecs = embedder.embed([c["text"] for c in chunks])

    # Build vector store with metadata for retrieval
    vs = InMemoryVectorStore(dim=vecs.shape[1])
    metadatas = []
    for i, c in enumerate(chunks):
        metadatas.append({
            "id": f"{call_id}:{i}",
            "call_id": call_id,
            "chunk_id": i,
            "start": c["start"],
            "end": c["end"],
        })
    vs.upsert(vecs, metadatas)

    # Local grading
    chunk_results = []
    for i, c in enumerate(chunks):
        det = deterministic_checks(expanded_texts[i])
        llm = grade_chunk_llm_like(expanded_texts[i])
        scores = combine_scores(det, llm)
        chunk_results.append({
            "chunk_id": c["chunk_id"],
            "start": c["start"], "end": c["end"],
            "scores": scores,
            "deterministic": det,
        })

    return {
        "chunks": chunks,
        "expanded_texts": expanded_texts,
        "embeddings": vecs,
        "vectorstore": vs,
        "chunk_results": chunk_results,
    }

def reduce_phase(call_id: str, map_out: Dict[str, Any]):
    """Aggregate chunk-level results into call-level metrics and rule checks."""
    chunk_results = map_out["chunk_results"]
    # Aggregate compliance: did any chunk contain disclosure?
    any_disclosure = any(cr["deterministic"]["disclosure_present"] for cr in chunk_results)
    pci_present = any(cr["deterministic"]["pci_risk"] for cr in chunk_results)

    # Weighted averages
    comp = np.mean([cr["scores"]["compliance"] for cr in chunk_results])
    qual = np.mean([cr["scores"]["quality"] for cr in chunk_results])
    eng = np.mean([cr["scores"]["engagement"] for cr in chunk_results])

    # Pattern: disclosures only at end (e.g., last 20% of chunks)
    last_k = max(1, int(0.2 * len(chunk_results)))
    disclosures_positions = [i for i, cr in enumerate(chunk_results) if cr["deterministic"]["disclosure_present"]]
    disclosure_late_pattern = (
        len(disclosures_positions) > 0 and
        all(pos >= len(chunk_results) - last_k for pos in disclosures_positions)
    )

    rules = {
        "required_disclosure_made": bool(any_disclosure),
        "pci_risk_detected": bool(pci_present),
        "disclosure_only_at_end": bool(disclosure_late_pattern),
    }

    call_metrics = {
        "compliance_avg": round(float(comp), 3),
        "quality_avg": round(float(qual), 3),
        "engagement_avg": round(float(eng), 3),
    }

    return {"rules": rules, "call_metrics": call_metrics}

def refine_phase(reduce_out: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a balanced scorecard and assign a final verdict + coaching notes."""
    m = reduce_out["call_metrics"]
    rules = reduce_out["rules"]
    # Balanced score: heavier weight on compliance
    final_score = 0.5 * m["compliance_avg"] + 0.3 * m["quality_avg"] + 0.2 * m["engagement_avg"]

    verdict = "Pass"
    reasons = []

    if not rules["required_disclosure_made"]:
        verdict = "Audit Required"
        reasons.append("Missing required disclosure.")
    elif rules["pci_risk_detected"]:
        verdict = "Audit Required"
        reasons.append("Potential PCI risk detected.")
    elif rules["disclosure_only_at_end"] or m["quality_avg"] < 0.7:
        verdict = "Needs Coaching"
        if rules["disclosure_only_at_end"]:
            reasons.append("Disclosures consistently only at the end.")
        if m["quality_avg"] < 0.7:
            reasons.append("Quality below target threshold.")

    notes = []
    if rules["disclosure_only_at_end"]:
        notes.append("Introduce disclosures earlier in the call flow.")
    if m["engagement_avg"] < 0.8:
        notes.append("Increase active listening and empathy statements.")
    if m["quality_avg"] < 0.8:
        notes.append("Use concise confirmations and avoid overlong explanations.")

    return {
        "final_score": round(float(final_score), 3),
        "verdict": verdict,
        "reasons": reasons,
        "coaching_notes": notes,
    }

def run_pipeline(call_id: str, transcript: str) -> Dict[str, Any]:
    m = map_phase(call_id, transcript)
    r = reduce_phase(call_id, m)
    f = refine_phase(r)
    report = {
        "call_id": call_id,
        "metrics": r["call_metrics"],
        "rules": r["rules"],
        "refine": f,
        "chunks": m["chunks"],
        "chunk_results": m["chunk_results"],
    }
    return report