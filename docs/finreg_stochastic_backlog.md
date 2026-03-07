# FinReg Stochastic Backlog (Shadow-Only)

Bu backlog, baseline karar hattini bozmadan stochastic yeniligi ilerletmek icindir.

## Sprint Goal

1. Shadow modda yeni stochastic epistemic adayini dogrulamak.
2. Baseline'i gecmeyen adayi hizla elemek.
3. Sadece kazanan adayi kontrollu aktivasyona tasimak.

## Guardrails (Locked)

1. Active gate karari baseline'dan cikacak.
2. Stochastic skorlar sadece loglanacak (shadow).
3. 20Q elemeden gecmeyen aday 50Q'ya cikmayacak.

## Task List

1. `Task-S1`: Shadow adapter katmani
   - `u_epi_source` secilebilir olacak: `logit_mi`, `stochastic_ou`, `stochastic_langevin`.
   - Output: per-query `u_epi_baseline`, `u_epi_stochastic`.
   - Done kriteri: eval JSONL icinde iki alan da dolu.

2. `Task-S2`: OU first candidate (shadow)
   - OU tabanli ornekleme modulu ekle.
   - Baseline ile ayni input/soru setinde shadow skor uret.
   - Done kriteri: 20Q kosusu hata vermeden tamamlanir.

3. `Task-S3`: Langevin second candidate (shadow)
   - Langevin/ULA tabanli ornekleme modulu ekle.
   - Ayni protokolde shadow skor uret.
   - Done kriteri: 20Q kosusu hata vermeden tamamlanir.

4. `Task-S4`: Standard compare raporu
   - Ayni formatta tablo:
   - `answered_contradiction_rate`, `abstain_rate`, `retrieve_more_rate`, `ECE/Brier` (varsa), runtime.
   - Done kriteri: tek JSON + tek md ozet.

5. `Task-S5`: 20Q gate criterion
   - Gecis kriteri:
   - `answered_contradiction_rate <= baseline`
   - `abstain_rate` kabul bandinda
   - Runtime maliyeti kabul edilebilir.
   - Done kriteri: PASS/FAIL etiketi.

6. `Task-S6`: 50Q confirm (sadece PASS aday)
   - Sadece 20Q PASS adayi 50Q seed=7 kos.
   - Done kriteri: 50Q rapor dosyasi.

7. `Task-S7`: seed robustness (sadece 50Q PASS aday)
   - seed=11 ve seed=19.
   - Done kriteri: mean/std ozet tablosu.

8. `Task-S8`: activation proposal (opsiyonel)
   - Sadece kriterleri gecerse aktivasyon onerisi yaz:
   - `shadow -> retrieve_more-only activation -> full activation`.
   - Done kriteri: rollout notu.

9. `Task-S9`: Mirror-Langevin adapter (shadow)
   - `--shadow-epi-source stochastic_mirror_langevin` aktif olsun.
   - 20Q seed=7 kosu.
   - Done kriteri: summary + per-question dump yazilir.

10. `Task-S10`: Wright-Fisher adapter (shadow)
   - `--shadow-epi-source stochastic_wright_fisher` aktif olsun.
   - 20Q seed=7 kosu.
   - Done kriteri: summary + per-question dump yazilir.

11. `Task-S11`: OU/Langevin/Mirror/WF compare
   - Tek tabloda 4 adayin `abstain_rate`, `answered_contradiction_rate`, `u_epi_stochastic_mean`.
   - Done kriteri: kisa karar notu (go/no-go).

12. `Task-S12`: 50Q promotion (yalnizca en iyi 1 aday)
   - Yalnizca kazanan aday 50Q'ya ciksin.
   - Done kriteri: 50Q summary ve PASS/FAIL etiketi.

## PASS/FAIL Rule (Single Source)

1. PASS:
   - `answered_contradiction_rate` baseline'dan iyi veya esit.
   - Coverage/abstain kabul bandinda.
   - En az 2 seed tutarli.

2. FAIL:
   - Yukaridaki kosullardan biri saglanmiyorsa aday dondurulur.

## Output Files

1. `evaluation_results/auto_eval/finreg_stochastic_shadow_20_seed7_*.json`
2. `evaluation_results/auto_eval/finreg_stochastic_shadow_50_seed7_*.json`
3. `docs/finreg_stochastic_compare.md`
4. `docs/finreg_stochastic_activation_note.md` (yalnizca PASS durumunda)

## Expanded Stochastic Candidate Bank (Research Update)

Bu bolum, FinReg odakli epistemic yenilik icin "denemeye deger" stokastik surec adaylarini genisletir.
Hedef: baseline'i bozmadan shadow hattinda hizli eleme yapmak.

### Tier-A (ilk denenecek, en mantikli)

1. `Ornstein-Uhlenbeck (OU)` (mevcut)
   - Durum: calisiyor, dengeli shadow davranis.
   - Rol: dusuk riskli referans.

2. `Langevin / ULA` (mevcut)
   - Durum: calisiyor, daha konservatif.
   - Rol: agresif epistemic artisi icin kontrol noktasi.

3. `Mirror-Langevin (MALA/MLA constrained)`
   - Neden: constrained/support-aware sampling (policy simplexi ve bounded skorlar icin uyumlu).
   - Beklenti: OU/Langevin'e gore daha stabil ve geometriye uyumlu skor.

4. `Wright-Fisher / simplicial diffusion`
   - Neden: simplex geometriyi dogrudan modelliyor (3-class NLI posterioru icin dogal).
   - Beklenti: epistemic skorda daha fiziksel/sinirli davranis.

### Tier-B (ikinci dalga, uygulanabilir ama orta maliyet)

1. `SGHMC / thermostat family`
   - Neden: momentum + friction ile daha iyi karisim, klasik SGLD zayifliklarini azaltir.
   - Beklenti: daha iyi calibration/epistemic ayrimi.

2. `Stochastic Gradient Barker Dynamics (SGBD)`
   - Neden: hiperparametre hassasiyetine daha robust olma iddiasi.
   - Beklenti: threshold tuning maliyetini azaltabilir.

3. `Reflected / Primal-Dual / Proximal Langevin (constrained)`
   - Neden: hard constraints altinda sampling teorisi guclu.
   - Beklenti: outlier skor patlamalarini azaltma.

4. `Subspace Langevin Monte Carlo (SLMC)`
   - Neden: ill-conditioned ortamlarda daha verimli sampling.
   - Beklenti: benzer kaliteyi daha dusuk runtime ile yakalama.

### Tier-C (arastirma/deep future work)

1. `PDMP samplers (Bouncy/Zig-Zag family for BNN posterior)`
   - Neden: non-reversible dinamiklerle mixing hizi avantajlari.
   - Risk: implementasyon karmaşigi yuksek.

2. `SMC-tempered Bayesian samplers`
   - Neden: calibration ve epistemic gucunde guclu sonuclar.
   - Risk: runtime/pipeline maliyeti yuksek.

3. `Schrodinger/Follmer bridge tabanli posterior transport`
   - Neden: transport-perspektifi ile hizli posterior yaklastirma potansiyeli.
   - Risk: RAG entegrasyonu erken asama.

## Recommended Execution Order (FinReg-only)

1. `Tier-A` tamamla: `OU -> Langevin -> Mirror-Langevin -> Wright-Fisher`
2. Her aday icin sadece `20Q shadow seed=7` (hizli eleme)
3. PASS adaylar icin `50Q shadow seed=7`
4. Sadece en iyi 1-2 aday icin `seed=11,19` robustness
5. Active gate'e gecis yok: sadece shadow compare dokumani uret

## Fast PASS Gate (20Q shadow)

1. `answered_contradiction_rate <= OU`
2. `abstain_rate` kabul bandinda (OU etrafinda)
3. `u_epi_stochastic_mean` ayirt edici ama asiri degil (all-answer collapse olmamali)
4. Runtime kabul edilebilir (OU'ya gore dramatik artis yok)

## Notes on Novelty Claim (pragmatic)

1. Literaturde SG-MCMC ve constrained sampling guclu, fakat FinReg-RAG gating'te
   stochastic epistemic adapter kullaniminin dogrudan yaygin bir kalibi net degil.
2. Bu nedenle iddia sekli:
   - "RAG uncertainty problemine stochastic process tabanli shadow epistemic adapter"
   - "active gate'i bozmadan kontrollu entegrasyon"

## Research Sources (primary papers)

1. SGHMC: https://arxiv.org/abs/1402.4102
2. Stochastic gradient thermostats (NIPS 2014): https://proceedings.neurips.cc/paper_files/paper/2014/hash/b610047c85e73cb7ec04fd36ec503f93-Abstract.html
3. CCAdL thermostat: https://arxiv.org/abs/1510.08692
4. SGBD (AISTATS 2024): https://arxiv.org/abs/2405.08999
5. Mirror Langevin (ALT 2022): https://proceedings.mlr.press/v167/li22b.html
6. Metropolis-adjusted Mirror Langevin (COLT 2024): https://proceedings.mlr.press/v247/srinivasan24a.html
7. Bregman Proximal LMC (ICML 2022): https://proceedings.mlr.press/v162/lau22a.html
8. ProxMCMC constrained/regularized: https://arxiv.org/abs/2205.07378
9. Primal-Dual LMC constrained sampling (NeurIPS 2024): https://arxiv.org/abs/2411.00568
10. Reflected Langevin convergence (2026): https://arxiv.org/abs/2512.00386
11. Subspace LMC (2025): https://arxiv.org/abs/2412.13928
12. PDMP for Bayesian NNs (rev. 2025): https://arxiv.org/abs/2302.08724
13. SMC+SGHMC tempered deep ensembles (2025): https://arxiv.org/abs/2505.11671
14. Scalable Bayesian Monte Carlo (2025): https://arxiv.org/abs/2505.13585
15. Wright-Fisher unification for discrete/gaussian/simplicial diffusion (2025): https://arxiv.org/abs/2512.15923
