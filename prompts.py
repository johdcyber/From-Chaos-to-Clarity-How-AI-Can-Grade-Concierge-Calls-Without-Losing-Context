# Example prompts/rubrics for real LLM calls. Adapt to your model of choice.
CHUNK_RUBRIC = """
You are grading a chunk from a long phone call. You will receive the target chunk
and its immediate neighbors for context. Provide a JSON object with the following keys:

- "disclosure_present": boolean (True if the chunk contains a required disclosure)
- "tone": float in [0,1] (polite, professional)
- "empathy": float in [0,1] (acknowledges and aligns with customer needs)
- "clarity": float in [0,1] (concise, unambiguous)
- "evidence": short phrases citing the sentences that justify your scores

Important:
- Consider the neighbor context to avoid mis-grading boundary sentences.
- Only mark "disclosure_present" if a clear, explicit disclosure is present.
Return only the JSON, no additional commentary.
"""