# External FinReg Dataset Gap

## Purpose

This note explains why existing open finance / regulation QA datasets do not fully match the current project target.

The project target is not generic finance QA. It is closer to:

- prudential / supervisory financial regulation
- multi-source evidence comparison
- ambiguity-aware answering
- grounding-aware gating
- answer vs retrieve-more vs abstain behavior

## Current Task Shape

The current finreg evaluation target is best described as:

- **high-stakes finreg ambiguity**
- **multi-regulator comparison**
- **prudential interpretation under partial conflict**

Typical project questions are not just:

- "What does document X say?"

They are often:

- "Do BCBS and regional supervisors differ on expectation Y?"
- "Are these requirements aligned or only partially aligned?"
- "Should the system answer, retrieve more evidence, or abstain?"

That is a narrower but more specialized task than most public datasets cover.

## External Datasets Reviewed

### 1. GBS-QA

Reference:

- https://aclanthology.org/2021.econlp-1.3.pdf

What it is:

- Banking standards QA dataset built from BCBS standards and FAQ material.

Why it is relevant:

- It is genuinely regulatory and banking-domain specific.
- BCBS content overlaps with some project themes.

Why it is not enough:

- It is mostly **single-source**.
- It is closer to **standard / FAQ clarification** than multi-regulator ambiguity.
- It does not target abstain / retrieve-more behavior.

Best description:

- **nearby task**
- not the same task

### 2. ObliQA / RIRAG-related regulatory QA

References:

- https://arxiv.org/abs/2409.05677
- https://github.com/RegNLP/ObliQADataset

What it is:

- Large regulatory QA / obligation-style dataset built from ADGM regulatory documents.

Why it is relevant:

- Uses real regulatory documents.
- Stronger retrieval and long-document setting than many finance QA sets.

Why it is not enough:

- It is centered on **regulatory obligation / compliance interpretation**.
- It is not primarily about **prudential supervisory disagreement across regulators**.
- It does not directly target ambiguity-aware gating.

Best description:

- **closest open neighbor**
- still different from the project target

### 3. FinTextQA

Reference:

- https://aclanthology.org/2024.acl-long.328/

What it is:

- Long-form finance QA benchmark built from finance textbooks and government agency websites.

Why it is relevant:

- Finance-domain LFQA.
- Some policy / government material appears in the source mix.

Why it is not enough:

- It is **general finance QA**, not prudential supervision QA.
- It is not built around regulator conflict or abstention behavior.
- It is broader, but less aligned to the exact project need.

Best description:

- **finance QA benchmark**
- not a finreg ambiguity benchmark

### 4. Single-regulator regulatory QA sets

Example pattern:

- Hugging Face or internal-style datasets built from one regulator or one handbook family

Why they are relevant:

- They confirm there is interest in regulatory QA.

Why they are not enough:

- Often **factual** or **single-source**.
- Usually not ambiguity-oriented.
- Usually do not encode answer / retrieve-more / abstain evaluation.

Best description:

- **regulatory QA exists**
- but mostly not in the target form needed here

## Core Gap

The current open-data gap is not:

- "there are no regulation datasets at all"

The real gap is:

- there is no obvious standard open dataset for
  - **multi-regulator prudential ambiguity**
  - **evidence conflict in finreg**
  - **grounding-aware abstention / retrieval escalation**

That means current public datasets are useful as:

- supporting evidence that nearby tasks exist
- retrieval / QA baselines
- domain adaptation helpers

But they are not enough as the main benchmark for this project.

## Implication For This Repo

The current internal finreg set should be understood as:

- an **internal stress-test slice**
- not a standard public benchmark

Its current value is that it targets a real evaluation gap:

- ambiguity under prudential supervision
- regulator comparison
- cautious behavior under partial support

Its current weakness is that its provenance and construction method are not yet documented strongly enough.

## Practical Conclusion

The project suspicion was broadly correct:

- open finance / regulation QA datasets exist
- but they do **not** cleanly match the current target

The right conclusion is not:

- "public datasets are irrelevant"

The right conclusion is:

- "public datasets are adjacent, but the project target is narrower and more specialized"

That is why the repo likely needs:

1. a documented internal finreg ambiguity slice
2. explicit question taxonomy
3. coverage labels tied to the real corpus
4. separate comparison between:
   - factual regulatory QA
   - obligation / compliance QA
   - prudential ambiguity / gating QA
