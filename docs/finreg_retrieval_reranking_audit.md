# FinReg Retrieval and Reranking Audit

This document records the current maturity assessment for the retrieval and reranking stack
under the active FinReg baseline.

Canonical baseline:

- `config/gating_finreg_ebcar_logit_mi_sc009.yaml`

## Active Retrieval Stack

- chunking: `section_aware`
- retrieval embedding: `sentence-transformers/all-MiniLM-L6-v2`
- vector store: `chroma`
- first-pass retrieval depth: `k = 20`
- reranker: `ebcar`
- reranker output depth: `top_k = 5`
- source-family balancing: enabled in `rag_pipeline`

## Code Paths

- retriever:
  - `src/rag/retriever.py`
- vector store search:
  - `src/vector_stores/stores/chroma_store.py`
- reranker:
  - `src/reranking/rerankers/ebcar_reranker.py`
- final balancing and context selection:
  - `src/rag/rag_pipeline.py`

## What Is Good Enough

### 1. Retrieval depth is no longer the main blocker

The active path retrieves `20` candidates before reranking. That is materially better than the
earlier `k=5` setup and is now large enough for meaningful reranking and family balancing.

### 2. Source-family balancing is useful

The pipeline now explicitly detects regulator names in the query and ensures that final context
selection can include multiple named source families.

This is important for:

- `BCBS vs ECB`
- `BCBS vs EBA`
- `ECB vs PRA`

style comparison questions.

### 3. `section_aware` chunking improved evidence quality

The move away from fixed-size chunking made the retrieval substrate more defensible.
This matters because reranking quality is limited by chunk quality.

## What Is Still Weak

### 1. Retriever is still a plain dense retriever

Current retriever behavior is simple:

- embed the query once
- search Chroma by vector similarity
- apply a score threshold

What it does not do:

- lexical retrieval
- hybrid dense + sparse fusion
- metadata-aware filtering
- query-type-specific retrieval strategy
- document-family-aware recall boosting

This means the first-pass recall is still relatively naive for regulatory questions that depend
on headings, exact phrases, or institution-specific terminology.

### 2. `EBCAR` is still heuristic, not a strong learned reranker

`EBCAR` combines:

- semantic similarity
- original retriever score
- position
- evidence count / confidence
- length
- title overlap

This is useful, but it is still a hand-weighted feature combiner.

What it does not do:

- joint query-passage scoring at token level
- cross-attention style relevance reasoning
- direct learned discrimination between subtle supervisory interpretations

This is why `EBCAR` should be treated as a practical lightweight reranker, not as a high-end
relevance model.

### 3. Final context selection mixes retrieval and policy logic

The same pipeline layer currently handles:

- retrieval
- reranking
- family balancing
- gating preparation

This works, but it makes diagnosis harder. Some issues that look like retrieval issues are
actually post-rerank selection issues.

### 4. Current embedding choice is serviceable, not strong

The active embedding model is:

- `sentence-transformers/all-MiniLM-L6-v2`

It is fast and stable, but it is not a strong domain-aware retrieval model.
For high-stakes prudential comparison questions, it is likely only a baseline-tier choice.

## Current Maturity Judgment

### Retrieval

- status: `adequate but not mature`

Reason:

- good enough to support a working baseline
- not good enough to be considered corpus-level mature

### Reranking

- status: `useful but not mature`

Reason:

- `EBCAR` is clearly helping
- but it remains heuristic and limited on fine-grained supervisory divergence

### Combined retrieval + reranking stack

- status: `below corpus maturity`

This stack is operational, but it has not yet reached the same confidence level as the corpus.

## Main Risks

### 1. False divergence from weak first-pass recall

If one regulator family is simply not recalled strongly enough, the system may appear to detect
cross-regulator disagreement when the real problem is incomplete evidence.

### 2. Heuristic reranker bias

Title overlap and semantic similarity can over-favor documents with surface alignment while
missing the more policy-relevant passage.

### 3. Hard questions still mix retrieval failure and question hardness

Some `phase15_hard` questions are genuinely difficult.
Others are hard because the current retrieval stack is still too weak.

Without careful diagnosis, those two failure modes get conflated.

## Recommended Next Sequence

### 1. Build a question-level retrieval/rerank error map

For current hard questions, record:

- whether the right regulator families appear in first-pass recall
- whether the right documents survive reranking
- whether final top-5 selection drops useful evidence

This should be done before changing models.

### 2. Evaluate first-pass recall separately from rerank quality

For a small hard-question set, compare:

- top-20 retrieved candidates
- top-5 after reranking
- top-5 after family balancing

This isolates where the loss happens.

### 3. Only then decide the next intervention

Priority order:

1. reranker / final selection tuning
2. retrieval strategy improvement
3. embedding model swap

Not the other way around.

## Working Conclusion

The current retrieval and reranking stack is:

- strong enough for a defensible baseline
- weak enough that it should not yet be treated as corpus-level mature

The next work should therefore focus on **retrieval/reranking failure analysis**, not blind model
swaps or broad experimentation.
