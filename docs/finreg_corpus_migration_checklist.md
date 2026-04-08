# FinReg Corpus Migration Checklist

## Why This Exists

The current finreg stack still assumes a synthetic bootstrap corpus in several places.

Before switching to a real finreg corpus, these references and assumptions must be updated in a controlled way.

## Current Synthetic Assumptions

### Corpus builder

- `scripts/build_high_stakes_bootstrap_corpora.py`

This script writes:

- `data/corpora/finreg_corpus.jsonl`

This output is synthetic and should no longer be treated as canonical once the real corpus exists.

### Indexing docs

These docs currently point to the synthetic JSONL corpus:

- `docs/high_stakes_domain_set.md`
- `docs/baseline_locked.md`

### Runtime config

Many finreg configs point to the same vector DB location:

- `./data/vector_db/domain_finreg`

This is acceptable, but only after the collection has been rebuilt from real processed artifacts.

## Target Migration Shape

### Old path

```text
synthetic JSONL -> index_domain_corpus.py -> data/vector_db/domain_finreg
```

### New path

```text
raw finreg docs -> processed finreg artifacts -> chunk artifacts -> index build -> data/vector_db/domain_finreg
```

## Script-Level Impact

### Keep

- `scripts/index_domain_corpus.py`
- `scripts/eval_grounding_proxy.py`
- `scripts/run_domain_questions.py`

These scripts can remain in the workflow, but their inputs should ultimately depend on the real corpus and rebuilt index.

### Re-scope

- `scripts/build_high_stakes_bootstrap_corpora.py`

This should become explicitly:

- synthetic-only
- bootstrap-only
- not canonical finreg evidence

### Add or adapt

Needed next:

1. raw finreg ingestion/build script
2. processed artifact builder
3. question coverage audit script

## Documentation Updates Required

Once the real corpus exists, update:

- `docs/high_stakes_domain_set.md`
- `docs/baseline_locked.md`
- `data/domain_finreg/README.md`

These files currently create the impression that the bootstrap corpus is the actual domain corpus.

## Runtime Validation After Migration

After rebuilding the real finreg corpus:

1. rebuild `data/vector_db/domain_finreg`
2. confirm index count is much larger than the current synthetic 23-document setup
3. run finreg 20Q sanity eval
4. inspect answered examples for source grounding quality
5. only then revisit detector/gating conclusions

## Decision Rule

Do not carry forward old finreg detector or gating conclusions as-is if they were obtained on the synthetic corpus.

Those results should be treated as:

- bootstrap proxy findings

not:

- final finreg system evidence
