# ASQA Knowledge Corpus

Build a JSONL corpus from ASQA knowledge passages (Wikipedia snippets embedded in ASQA).
Use this while AmbigQA evidence is downloading.

## Build corpus

```bash
PYTHONPATH=. venv/bin/python scripts/prepare_asqa_corpus.py \
  --input data/ambiguity_datasets/03_asqa/dataset/ASQA.json \
  --output data/corpora/asqa_wiki_knowledge.jsonl \
  --target-docs 20000 \
  --max-chars 1200 \
  --min-chars 200
```

Optional: include QA-pairs as extra docs (not pure Wikipedia evidence):

```bash
PYTHONPATH=. venv/bin/python scripts/prepare_asqa_corpus.py \
  --include-qa-pairs
```

## Index

```bash
PYTHONPATH=. venv/bin/python - <<'PY'
from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline('config/gating_demo.yaml')
count = pipeline.index_documents('data/corpora/asqa_wiki_knowledge.jsonl')
print('indexed', count)
PY
```
