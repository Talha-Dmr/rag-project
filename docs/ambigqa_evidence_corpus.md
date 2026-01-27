# AmbigQA Evidence Corpus

This uses the AmbigQA evidence dataset (Wikipedia-derived articles) to build a JSONL corpus
for the RAG index.

## Download

The evidence zip is linked in `data/ambiguity_datasets/02_ambigqa/README.md`.
Place the zip in `data/ambiguity_datasets/02_ambigqa/` and extract it there.

Expected files after extract:
- `train_with_evidence_articles.json`
- `dev_with_evidence_articles.json`
- `test_with_evidence_articles_without_answers.json`

## Build corpus (20–50k docs)

```bash
PYTHONPATH=. venv/bin/python scripts/prepare_ambigqa_evidence_corpus.py \
  --input data/ambiguity_datasets/02_ambigqa/train_with_evidence_articles.json \
  --output data/corpora/ambigqa_wiki_evidence.jsonl \
  --target-docs 40000 \
  --max-articles-per-item 3 \
  --max-chars 1200 \
  --min-chars 200
```

## Index

```bash
PYTHONPATH=. venv/bin/python - <<'PY'
from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline('config/gating_demo.yaml')
count = pipeline.index_documents('data/corpora/ambigqa_wiki_evidence.jsonl')
print('indexed', count)
PY
```

Notes:
- This corpus is a Wikipedia subset aligned to AmbigQA evidence.
- Chunk size is controlled by `--max-chars`/`--min-chars`.
- If you only have the zip, pass `--input ...zip --member train_with_evidence_articles.json`.
