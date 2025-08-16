from typing import Dict, List, Tuple
import re
import math

DISCLOSURE_PATTERNS = [
    r"not guaranteed",
    r"subject to availability",
    r"cancellation (fees|policy|policies)",
]

PCI_PATTERNS = [
    r"\b\d{13,19}\b",  # naive credit card number detection
]

def deterministic_checks(text: str) -> Dict:
    """Regex-based checks for must-have phrases and PCI risk signals."""
    text_l = text.lower()
    disclosures = {pat: bool(re.search(pat, text_l)) for pat in DISCLOSURE_PATTERNS}
    pci_flags = {pat: bool(re.search(pat, text_l)) for pat in PCI_PATTERNS}
    return {
        "disclosures": disclosures,
        "pci_flags": pci_flags,
        "disclosure_present": any(disclosures.values()),
        "pci_risk": any(pci_flags.values()),
    }

def grade_chunk_llm_like(text: str) -> Dict:
    """LLM-like stub grader: heuristic scores for tone, empathy, and clarity (0-1).
    Replace this with calls to your LLM using prompts in `prompts.py`."""
    t = text.lower()
    # naive heuristics only for demo:
    empathy = 1.0 if any(w in t for w in ["understood", "sure", "glad", "sorry", "help"]) else 0.4
    clarity = 1.0 if len(text.split()) < 180 else 0.6
    tone = 0.9 if "please" in t or "thanks" in t else 0.6
    return {
        "tone": tone,
        "empathy": empathy,
        "clarity": clarity,
    }

def combine_scores(det_checks: Dict, llm_scores: Dict) -> Dict:
    """Combine deterministic and LLM-like signals into a chunk scorecard."""
    compliance_score = 1.0 if det_checks["disclosure_present"] else 0.5
    if det_checks["pci_risk"]:
        compliance_score = min(compliance_score, 0.2)  # heavy penalty if PCI risk detected
    quality_score = 0.5 * llm_scores["tone"] + 0.5 * llm_scores["clarity"]
    engagement_score = llm_scores["empathy"]
    return {
        "compliance": round(compliance_score, 3),
        "quality": round(quality_score, 3),
        "engagement": round(engagement_score, 3),
    }