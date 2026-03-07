# FinReg Execution Plan (Scope-Frozen)

Bu dosya, mevcut faz için tek kaynak planidir. Amaç dagilmadan ilerlemek ve her adimda karar kriteriyle gitmektir.

## 0) Frozen Decisions

1. Domain: sadece `FinReg`.
2. Aktif karar hatti (production-like): `gating_finreg_ebcar_logit_mi_sc009`.
3. `rep_mi` bu fazda elendi (20Q smoke testte abstain coktu).
4. `health/disaster` bu fazda disarida.
5. Stochastic epistemic (OU/Langevin vb.) iptal degil; ayri/shadow track.

## 1) Sprint Objective

Baseline'i bozmadan iki-kanalli uncertainty gecisine hazir olmak:

1. `U_epi` (mevcut epistemic) korunacak.
2. `U_ale` (ayrik aleatoric skor) shadow modda eklenecek.
3. Kriter gecilirse 2D policy aktivasyonu dusunulecek.

## 2) Phase A - Baseline Lock

### A1) Baseline run (single source of truth)

```bash
cd /home/talha/projects/rag-project && env PYTHONPATH=. HF_HOME=./models/llm TRANSFORMERS_CACHE=./models/llm venv312/bin/python -u scripts/eval_grounding_proxy.py --config gating_finreg_ebcar_logit_mi_sc009 --questions data/domain_finreg/questions_finreg_conflict_50.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/finreg_baseline_locked.json
```

### A2) Baseline summary check

```bash
cat evaluation_results/auto_eval/finreg_baseline_locked.json
```

### A3) Baseline KPIs (freeze)

1. `abstain_rate`
2. `stats_all.contradiction_rate`
3. `stats_answered.contradiction_rate`
4. `stats_all.source_consistency`
5. `stats_all.retrieval_mean_score`

## 3) Phase B - Aleatoric Split (Shadow Only)

Bu fazda gate karari degismez. Sadece ek sinyal hesaplanir ve loglanir.

### B1) New fields (per query)

1. `u_epi` (existing signal, e.g. logit_mi)
2. `u_ale` (composed score from conflict/consistency/retrieval ambiguity)
3. `action_actual` (current gate action)
4. `action_shadow_2d` (2D policy recommendation)

### B2) Minimum Aleatoric composition (first cut)

1. `contradiction_prob`
2. `1 - source_consistency`
3. retrieval spread (e.g. max - mean or std)

Not: Bu fazda sadece kayit/rapor, karar etkisi yok.

## 4) Phase C - 2D Policy in Shadow

Ilk policy:

1. high `u_epi`, low `u_ale` -> `retrieve_more`
2. low `u_epi`, high `u_ale` -> `abstain` (veya cautious)
3. high `u_epi`, high `u_ale` -> `abstain`
4. low `u_epi`, low `u_ale` -> `answer`

Goal: Shadow policy, riskli answered ornekleri baseline'dan iyi yakaliyor mu?

## 5) Phase D - Controlled Activation (Only If Criteria Pass)

Activation sadece su durumda:

1. `stats_answered.contradiction_rate` baseline'dan iyi.
2. `abstain_rate` kabul bandinda.
3. Davranis stabil.

Gecmezse 2D kapali kalir, baseline devam.

## 6) Separate Track - Stochastic Epistemic

Bu track ana hattan ayridir:

1. OU/Langevin adaylari shadow olarak denenir.
2. Kazanan olursa sonra gate epistemic kaynagina aday olur.
3. Ana FinReg baseline akisi bu track yuzunden bozulmaz.

## 7) Out of Scope (This Phase)

1. Multi-domain tuning.
2. Genis seed/domain sweeps.
3. Ayni anda birden fazla gate kurali degisikligi.

## 8) Deliverables

1. `docs/finreg_execution_plan.md` (this file)
2. `evaluation_results/auto_eval/finreg_baseline_locked.json`
3. shadow-calisma ciktilari (phase B/C)
4. kisa compare notu (baseline vs shadow policy impact)

## 9) Core Principle (Locked)

Baseline'in iyi cikmasi, yenilik hedefinden vazgecmek anlamina gelmez.

1. Ana hedef korunur: epistemic + aleatoric ayrimi ve stochastic uncertainty arastirmasi.
2. Uretim-benzeri karar hatti korunur: aktif gate kararini baseline verir.
3. Yenilik guvenli sekilde gelistirilir: shadow -> kanit -> kontrollu aktivasyon.
4. Baseline'i gecemeyen adaylar hizla dondurulur; amac sapmasi degil, metodolojik disiplin saglanir.
