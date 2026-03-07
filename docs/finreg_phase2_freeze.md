# FinReg Phase-2 Freeze Note

Bu not, baseline disi adaylarin neden donduruldugunu ve su anki karar durumunu sabitler.

## Scope

1. Domain: FinReg only
2. Active decision path: `gating_finreg_ebcar_logit_mi_sc009`
3. Eval slice: 20Q, seed=7 (hizli eleme)

## Baseline (reference)

Source: `evaluation_results/auto_eval/finreg_baseline_locked.json`

1. `actions`: `none=7`, `retrieve_more=13`
2. `abstain_rate`: `0.65`
3. `stats_answered.contradiction_rate`: `0.0571`

## Eliminated Candidate: `rep_mi`

Source: `evaluation_results/auto_eval/finreg_rep_mi_20_seed7.json`

1. `actions`: `retrieve_more=20`
2. `abstain_rate`: `1.0`
3. Conclusion: practical collapse (full abstain behavior), eliminated for this phase.

## Shadow 2D Policy (v1/v2) Status

v1 source: `evaluation_results/auto_eval/finreg_shadow_two_channel_20_seed7_summary.json`
v2 source: `evaluation_results/auto_eval/finreg_shadow_two_channel_v2_20_seed7_summary.json`

### v1

1. `shadow actions`: `answer=17`, `retrieve_more=3`
2. `shadow abstain_rate`: `0.0`
3. `shadow answered_contradiction_rate`: `0.47`
4. Baseline'a gore belirgin kotu.

### v2 (conflict-aware u_ale)

1. `shadow actions`: `abstain=10`, `answer=10`
2. `shadow abstain_rate`: `0.5`
3. `shadow answered_contradiction_rate`: `0.54`
4. Baseline'a gore yine belirgin kotu.

## Decision (Locked)

1. Baseline active path olarak kalir.
2. `rep_mi` ve 2D shadow policy bu fazda dondurulur.
3. Stochastic/2D track iptal degil, shadow backlog olarak surdurulur.
4. Yeni aday sadece su kriterle yeniden aktive edilir:
   - `stats_answered.contradiction_rate <= baseline`
   - coverage/abstain kabul bandinda
   - en az 2 seed tutarlilik
