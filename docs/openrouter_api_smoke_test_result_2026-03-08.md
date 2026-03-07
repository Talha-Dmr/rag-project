# OpenRouter API Smoke Test Result (2026-03-08)

## Scope
- Goal: verify API-based RAG flow (without local LLM inference).
- Config used: `config/gating_finreg_openrouter_router_free.yaml`
- LLM route: `llm.type: openrouter` via `src/rag/openrouter_llm.py`

## Test Inputs
- Question file: `data/domain_finreg/questions_openrouter_smoke.jsonl`
- Questions:
  - What is model risk management?
  - List two key controls in model risk management.
  - What is the capital city of Japan?

## Execution Summary
- Date/time: 2026-03-08 00:10-00:20 (Europe/Istanbul)
- RAG indexing completed on a small quick corpus (`data/quick_finreg.txt`, local/ignored helper file).
- API call path worked with `model_name: openrouter/free`.
- Note: direct `qwen/qwen3-4b:free` hit HTTP 429 during this run (provider free-tier rate limit).
- Follow-up smoke run (2 queries) also completed successfully after cleanup.

## Sample Outputs (from smoke run)
- Q1: model risk management tanımı üretildi (beklenen alan içi yanıt).
- Q2: independent validation + governance gibi iki kontrol üretildi (beklenen alan içi yanıt).
- Q3 (alan dışı): `None` döndü.

## Eval JSON Summary
Command output summary from `scripts/eval_grounding_proxy.py` (3-question smoke set):

```json
{
  "total": 3,
  "abstain": 0,
  "abstain_rate": 0.0,
  "detector_failures": 3,
  "actions": {"none": 3},
  "u_epi_mean": 0.0,
  "u_ale_mean": 0.3333333333333333,
  "retrieval_spread_mean": 0.0
}
```

## Conclusion
- API-based integration is working.
- Repo was cleaned to keep API integration files and API smoke test artifacts only.
