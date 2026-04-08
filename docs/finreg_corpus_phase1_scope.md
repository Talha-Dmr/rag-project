# FinReg Corpus Phase-1 Scope

## Goal

Build the first real finreg corpus that is strong enough for:

- retrieval quality measurement
- source-grounded answer generation
- conflict-aware gating evaluation

This phase is intentionally narrower than "all financial regulation".

## Question Families The Corpus Must Support

The current finreg question sets cluster around these themes:

1. BCBS 239 / risk data aggregation
2. supervisory review and risk governance
3. data lineage and auditability
4. stress testing and prudential supervision
5. model risk management / validation
6. climate risk integration into ICAAP
7. outsourcing controls for risk systems
8. material reporting errors / remediation thresholds
9. audit trail retention
10. intraday liquidity monitoring
11. AI model explainability in regulated risk use cases
12. group / cross-border governance consistency

The phase-1 corpus should be selected to support these themes directly, not to maximize source count.

## In-Scope Source Families

### 1. BCBS / BIS

Priority: highest

Needed because BCBS is the anchor source in most sanity and conflict questions.

Target document families:

- BCBS 239 / risk data aggregation principles
- supervisory review / prudential governance principles
- stress testing principles or supervisory papers
- intraday liquidity monitoring principles
- climate risk principles relevant to banks

Expected role:

- global baseline source
- anchor side of most conflict comparisons

### 2. EBA

Priority: highest

Needed because many conflict questions compare BCBS with EU supervisory guidance.

Target document families:

- SREP or supervisory review guidance
- ICT / outsourcing / operational resilience guidance where it affects risk systems
- model risk / governance / internal control guidance
- climate risk supervisory guidance tied to ICAAP or governance
- reporting and data quality guidance if available

Expected role:

- EU supervisory comparator
- proportionality and implementation-detail source

### 3. PRA / Bank of England

Priority: high

Needed because several questions are really about local supervisory implementation, reporting controls, and operational expectations.

Target document families:

- supervisory statements on governance / model risk / international firms
- reporting instructions or operational reporting guidance
- statements touching data quality, controls, and remediation expectations
- climate-risk supervisory expectations

Expected role:

- UK implementation comparator
- operational and reporting-oriented source

### 4. ECB

Priority: medium-high

Needed because several conflict questions compare BCBS to a regional prudential supervisor with stronger implementation expectations.

Target document families:

- supervisory expectations on risk data, governance, stress testing, or ICAAP
- climate-risk supervisory expectations
- model governance / internal control expectations

Expected role:

- euro-area comparator
- acceleration / implementation pressure source

## Conditional Scope

### 5. Federal Reserve / OCC

Priority: conditional

These sources are useful for later conflict coverage, especially for:

- model risk management
- governance expectations
- AI / explainability and control discussions

But they should enter phase 1 only if they directly improve support for existing question families. Otherwise they should wait for phase 2.

## Out of Scope For Phase 1

- broad US multi-agency expansion
- MiCA / PSD2 / BRRD style horizontal EU legal coverage unless directly needed by active finreg questions
- academic papers
- news articles, commentary, blog posts
- synthetic benchmark notes as primary evidence

## Minimum Corpus Composition

Phase 1 should not ship unless it has:

- more than one source family
- multiple documents per source family
- real regulatory or supervisory documents
- enough coverage that at least the current 20Q seed set can be answered from the corpus

Practical minimum:

- BCBS: 4 to 6 documents
- EBA: 3 to 5 documents
- PRA/BoE: 3 to 5 documents
- ECB: 2 to 4 documents

This is still small, but materially better than the current 23-note synthetic corpus.

## Metadata Requirements

Each processed document/chunk should carry:

- `source_org`
- `document_family`
- `document_id`
- `title`
- `jurisdiction`
- `year`
- `document_type`
- `source_path`
- `page_start`
- `page_end`
- `section` if recoverable

## Acceptance Criteria

The phase-1 corpus is good enough to replace the synthetic bootstrap corpus only if:

1. It is reproducible from raw files and scripts.
2. The vector store can be rebuilt from processed artifacts.
3. The current finreg 20Q set is mostly answerable from corpus evidence.
4. Conflict questions can be grounded in real source divergence, not synthetic stance text.
5. The corpus is documented as canonical in one place.

## Next Step After Scope Freeze

After this scope is accepted:

1. create the document inventory list
2. define raw folder layout
3. build processed corpus pipeline
4. rebuild finreg vector DB
5. re-check question coverage and prune or rewrite weak questions
