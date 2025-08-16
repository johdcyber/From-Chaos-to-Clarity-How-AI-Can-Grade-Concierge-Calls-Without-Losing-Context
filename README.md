From Chaos to Clarity — A Developer’s Field Guide to Context‑Preserving QA for Long Calls

TL;DR
Grading long calls with an LLM is like judging a movie by four random screenshots—you miss the plot. The cure is context‑preserving chunking + vector retrieval and a Map → Reduce → Refine pipeline that treats local details and global narrative as first‑class citizens.

1) Problem, Intuition, and Why Your First Attempt Fails

Analogy: Imagine skimming a 300‑page novel by reading every 3rd page. You’ll catch witty lines but miss the whodunit. Long calls (> tens of thousands of tokens) force you to split the transcript; if you grade each chunk in isolation, you’ll misjudge the story:

Late disclosures: appear only near the end → “No disclosure found” (false negative).

Emotion arcs: frustration builds gradually → “Sentiment is neutral” (false).

Recovery: agent stumbles early but recovers late → “Poor call” (unfair).

Goal: Maximize recall of compliance moments while minimizing false alarms and keeping latency & cost reasonable.

2) Formal Problem Statement (with notation)

Let a call transcript be a token sequence 
𝑇
=
{
𝑡
1
,
…
,
𝑡
𝑛
}
T={t
1
	​

,…,t
n
	​

}.
We need a function 
𝐹
F that outputs a final grade 
𝐺
G with evidence and reasons:

𝐺
=
𝐹
(
𝑇
)
→
{
score
∈
[
0
,
1
]
,
verdict
∈
{
Pass
,
Coach
,
Audit
}
,
evidence
,
rationales
}
G=F(T)→{score∈[0,1],verdict∈{Pass,Coach,Audit},evidence,rationales}

Constraints:

LLM context limit 
𝐿
≪
𝑛
L≪n.

Grading must be explainable (point to spans) and scalable.

We decompose:

Map 
𝑓
f: chunk‑level scoring with neighbor context.

Reduce 
𝑔
g: global aggregation + rules & patterns.

Refine 
ℎ
h: balanced scorecard + verdict + coaching notes.

𝐺
=
ℎ
(
𝑔
(
{
𝑓
(
𝐶
𝑖
±
)
}
𝑖
)
)
G=h(g({f(C
i
±
	​

)}
i
	​

))

where 
𝐶
𝑖
±
C
i
±
	​

 is chunk 
𝑖
i expanded with 
±
± neighbors.

3) Architecture at 10,000 ft (Figures 1–3)

Figure 1 – The Chunking Problem: why naive splits lose meaning.

Figure 2 – Map → Reduce → Refine Pipeline: staged grading flow.

Figure 3 – Vector DB Integration for Contextual QA: embeddings store + retrieval loop that preserves continuity and supports cross‑call analytics.

4) Chunking Like a Pro (token‑aware + overlap + neighbor expansion)

Analogy: Cutting a song into 30‑second clips is fine—unless you cut right through the chorus. We prevent “chorus cutting.”

Practices

Token‑aware windows: size 500–1000 tokens (model‑dependent).

Overlap: 10–20% (e.g., 50–100 tokens) to reduce boundary truncation.

Neighbor expansion at grading time: include 
±
1
±1 adjacent chunk(s).

Boundary hints (optional, advanced):

Align splits to speaker turns, punctuation, or timecode seams.

Expand window if a sentence is split mid‑way.

Pseudocode

# sliding window with overlap
tokens = tokenize(transcript)
for start in range(0, len(tokens), chunk_size - overlap):
    end = min(start + chunk_size, len(tokens))
    yield Chunk(id=i, start=start, end=end, text=detok(tokens[start:end]))

def expand_with_neighbors(chunks, i, w=1):
    lo = max(0, i-w); hi = min(len(chunks)-1, i+w)
    return "\n".join(c.text for c in chunks[lo:hi+1])


The downloadable project implements this in chunker.py.

5) Embeddings & Vector Store (the librarian of your call universe)

Analogy: The vector DB is a librarian who files each paragraph by meaning, not just keywords. You can ask: “show me all disclosure‑ish passages from last week” or “give me neighbors around chunk 27.”

Key design

Embedding model: sentence embeddings, normalized.

Metadata per chunk: { call_id, chunk_id, start, end, speaker, timestamp }.

Indexing: upsert once; reuse for grading & analytics.

Retrieval patterns:

Neighbor fetch by chunk_id±1 to preserve continuity.

Semantic fetch by disclosure intent, objection patterns, etc.

Cross‑call queries for trends (agent/team/system‑wide).

Complexity & scale

Upsert: 
𝑂
(
𝑁
𝑑
)
O(Nd) where 
𝑁
N chunks, 
𝑑
d dims.

ANN retrieval: typically sub‑linear (HNSW/IVF; vendor‑specific).

You grade locally but avoid re‑embedding the world each time.

See embeddings.py and vectorstore.py (runnable in‑memory + Pinecone stub you can swap in).

6) Map: Local Grading (hybrid = rules + model)

Analogy: A pit crew: the rules inspector checks seatbelts & fuel (deterministic); the driver coach listens for smoothness (LLM).

Signals

Deterministic: regex for mandatory phrases, PCI/PII flags, policy keywords.

LLM/ML: subjective scores (tone, empathy, clarity), span evidence.

Stable JSON contract (from prompts.py):

{
  "disclosure_present": true,
  "tone": 0.86,
  "empathy": 0.82,
  "clarity": 0.91,
  "evidence": ["upgrades are not guaranteed", "cancellation fees may apply"]
}


Combining signals

# scoring.py
compliance = 1.0 if disclosure_present else 0.5
if pci_risk: compliance = min(compliance, 0.2)
quality = 0.5*tone + 0.5*clarity
engagement = empathy


Implemented in scoring.py. Replace the stub LLM with your favorite provider and keep JSON schema validation to avoid parsing surprises.

7) Reduce: Global Aggregation + Call‑Level Rules

Analogy: You’ve judged the scenes; now assess the whole film: Was the safety disclaimer shown at all? Was it hidden in the last 5 seconds?

Metrics

Averages: compliance_avg, quality_avg, engagement_avg.

Rules (examples you can extend):

required_disclosure_made (exists across any chunk).

disclosure_only_at_end (all disclosures fall in last 20% of chunks).

pci_risk_detected (any risky pattern).

Pseudocode

any_disclosure = any(cr.det.disclosure_present for cr in chunk_results)
last_k = max(1, int(0.2 * len(chunk_results)))
disclosure_positions = [i for i, cr in enumerate(chunk_results) if cr.det.disclosure_present]
late_only = len(disclosure_positions) > 0 and all(p >= len(chunk_results)-last_k for p in disclosure_positions)
rules = { "required_disclosure_made": any_disclosure, "disclosure_only_at_end": late_only, ... }


Concrete implementation in map_reduce_refine.py.

8) Refine: Balanced Scorecard, Verdicts, and Coaching Notes

Analogy: The restaurant review: you weigh hygiene (compliance) more than décor (tone), then write actionable tips.

Weighted score

final
=
0.5
⋅
compliance
+
0.3
⋅
quality
+
0.2
⋅
engagement
final=0.5⋅compliance+0.3⋅quality+0.2⋅engagement

Verdicts

Audit Required: missing disclosure or PCI risk.

Needs Coaching: disclosure late‑only or quality below threshold.

Pass: otherwise.

Notes (examples):

“Introduce disclosures earlier.”

“Use concise confirmations.”

“Increase active listening signals.”

Implemented in map_reduce_refine.py::refine_phase.

9) How to Build This Yourself (step‑by‑step with the included project)

Get the code → unzip call-qa-guide.zip

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # optional; demo runs without external deps
python main.py


Swap tokenization → use tiktoken (or model tokenizer).
Adjust chunk_size=800, overlap=80 (tune per model).

Swap embeddings → in embeddings.py, switch to SentenceTransformers or your hosted embeddings. Normalize vectors.

Drop in a vector DB → in vectorstore.py, replace InMemoryVectorStore with your hosted index. Keep metadata: {call_id, chunk_id, start, end}.

Call your LLM → in scoring.grade_chunk_llm_like, replace the stub with an API call using prompts.CHUNK_RUBRIC. Validate JSON against a schema.

Tune rules → edit thresholds in map_reduce_refine.py::refine_phase. Put them in config to A/B test.

Evaluate (see §11): label 50–200 calls, compute precision/recall of disclosures, false audit rate, and time‑to‑grade. Iterate on chunk size, overlap, thresholds.

10) Retrieval Recipes You’ll Use Weekly

Neighbor continuity (same call)
filter={"call_id": X} + fetch by chunk_id±1 from metadata.

Find all disclosures last week
Embed query “disclosure about cancellation or not guaranteed”, top_k=1000, filter by timestamp∈[t0,t1].
Aggregate by agent_id/team_id.

Explain a verdict
For each rule hit, include the chunk ids and evidence spans surfaced by the LLM/deterministic checks.

11) Evaluation Methodology (how to know it works)

Labeling: have reviewers mark (a) disclosure presence & timing, (b) objection handling, (c) tone snapshots.

Metrics:

Disclosure detection: precision, recall, F1.

Call verdicts: accuracy, ROC‑AUC (if probabilistic).

Scoring correlation: Spearman’s ρ vs. human scores.

Ablations (measure each improvement):

No overlap → +overlap → +neighbor expansion.

No Reduce rules → +Reduce rules.

No vector retrieval → +vector retrieval.

Confidence intervals: bootstrap 1k resamples for F1 ± CI.

12) Failure Modes & Fixes

LLM hallucination → force JSON schema, require evidence spans, reject/reattempt on invalid JSON.

Over‑penalizing harmless chunks → Reduce with call‑level logic; de‑emphasize isolated negatives.

Missed late disclosures → overlap + neighbor expansion; late‑only rule flags pattern.

Cost blow‑up → keep chunk sizes moderate; cache embeddings; ANN indexes; batch LLM calls.

13) Privacy, Security, and Risk

Redact PCI/PII before embedding.

Region‑bound storage, encryption at rest, access logs.

Configurable retention windows; delete or re‑embed on policy changes.

14) Scaling & Ops

Throughput: parallelize Map workers; Reduce runs when all chunk jobs complete (fan‑in).

Idempotency: deterministic ids {call_id}:{chunk_id}.

Observability: structured logs per chunk verdict; trace ids across Map→Reduce→Refine.

Complexity

Chunking: 
𝑂
(
𝑛
)
O(n).

Embedding: 
𝑂
(
𝑁
𝑑
)
O(Nd).

Retrieval: ANN sub‑linear.

LLM cost: 
𝑂
(
𝑁
)
O(N) calls; batch + cache where possible.

15) Extensions (when you want more)

Prosody: add acoustic features (pitch, energy) to detect stress/recovery arcs.

Adaptive chunk sizes: bigger windows for calm phases; smaller for dense compliance zones.

Graph stitching: connect chunks by semantic topic; walk the graph to assemble global narratives.

Appendix A — Minimal Pseudocode (end-to-end)
def grade_call(transcript, call_id):
    # 1) Chunk
    chunks = chunk_by_tokens(transcript, chunk_size=800, overlap=80)

    # 2) Embed + index (id = f"{call_id}:{i}")
    vecs = embeddings.embed([c.text for c in chunks])
    vector_db.upsert(vecs, [{"call_id": call_id, "chunk_id": i, "start": c.start, "end": c.end} for i, c in enumerate(chunks)])

    # 3) Map (local)
    results = []
    for i, c in enumerate(chunks):
        text_with_neighbors = expand_with_neighbors(chunks, i, w=1)
        det = deterministic_checks(text_with_neighbors)
        llm = llm_grade(text_with_neighbors)        # JSON with tone/empathy/clarity, evidence
        scores = combine(det, llm)
        results.append({"i": i, "det": det, "scores": scores})

    # 4) Reduce (global)
    rules, call_metrics = aggregate_and_apply_rules(results)

    # 5) Refine (final)
    verdict = finalize(call_metrics, rules)  # score, label, notes

    return { "call_id": call_id, "metrics": call_metrics, "rules": rules, "verdict": verdict, "chunks": results }


A full implementation (with runnable stubs) is already packaged in call-qa-guide.zip.

Why this process is great (and not just “nice”)

Context is preserved by design (overlap, neighbors, retrieval).

Accuracy improves because local signals are checked but global rules decide.

It scales: vector DB eliminates re‑processing; Map workers parallelize.

It’s explainable: every verdict cites exact chunks and evidence phrases.

It’s adaptable: swap models, tweak rules, add new labels without rewriting the world.
