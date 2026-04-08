# FinReg Phase-1.5 Expansion Shortlist

## Goal

Freeze the corpus at a **practical upper bound** for this project:

- not "all finreg"
- not a broad multi-jurisdiction regulatory dump
- just enough real prudential / supervisory coverage to support the
  hallucination-detector and stochastic-gating work

Target end state:

- about `30-50` official documents
- a few thousand chunks after indexing
- coverage centered on prudential / supervisory ambiguity, not general finance

Current state:

- about `18` official documents in the phase-1 corpus
- source families:
  - `BCBS`
  - `EBA`
  - `ECB`
  - `PRA/BoE`

This means phase-1.5 should add roughly `12-20` carefully selected documents.

## Selection Rule

Add documents only if they improve at least one of these:

- multi-regulator comparison coverage
- governance / remediation / escalation evidence
- model-governance evidence
- group-wide / cross-border supervisory evidence
- operational resilience / outsourcing controls for risk systems

Do not add:

- news
- commentary
- academic papers
- horizontal finreg material unrelated to the active prudential questions
- broad legal corpora just for volume

## Phase-1.5 Priority Additions

These are the highest-value additions for the current project.

### BCBS

1. **Corporate governance principles for banks**
   - Why:
     - strengthens governance, board responsibility, control-function, and group-governance comparisons
     - directly useful for EBA / ECB / PRA governance alignment questions
   - Official source:
     - https://www.bis.org/bcbs/publ/d328.htm

2. **Stress testing principles**
   - Why:
     - strengthens stress-testing and supervisory challenge coverage
     - improves the prudential governance side of capital / management-action questions
   - Official source:
     - https://www.bis.org/bcbs/publ/d450.htm

3. **Core Principles for effective banking supervision**
   - Why:
     - adds higher-level supervisory framing for remediation, escalation, and governance expectations
     - useful as an anchor for "what supervisors are expected to do" style questions
   - Official source:
     - https://www.bis.org/bcbs/publ/d573.htm

### EBA

4. **Guidelines on ICT and security risk management**
   - Why:
     - improves ICT / operational controls / resilience evidence
     - strengthens outsourcing and risk-systems governance coverage
   - Official source:
     - https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/internal-governance/guidelines-ict-and-security-risk-management

5. **Guidelines on recovery plan indicators**
   - Why:
     - directly useful for escalation, thresholds, remediation, and persistent-deficiency handling
     - strengthens questions that currently ask too broadly about supervisory follow-up
   - Official source:
     - https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/recovery-resolution-and-dgs/guidelines-recovery

6. **Guidance on management-body suitability / key function holders**
   - Why:
     - improves governance-accountability comparisons
     - supports questions on responsibility allocation and oversight quality
   - Official source:
     - https://www.eba.europa.eu/publications-and-media/press-releases/eba-and-esma-provide-guidance-assess-suitability-management

7. **Remuneration policy guidance**
   - Why:
     - adds governance and incentive-structure evidence
     - useful where risk culture and management-body incentives matter
   - Official source:
     - https://www.eba.europa.eu/regulation-and-policy/remuneration

### ECB

8. **Guide on governance and risk culture**
   - Why:
     - high-value addition for management-body effectiveness, internal control functions, risk culture, and remediation expectations
     - likely one of the best ECB-side comparators for governance questions
   - Official source:
     - https://www.bankingsupervision.europa.eu/framework/legal-framework/public-consultations/html/governance_and_risk_culture.en.html
   - Note:
     - this needs final URL validation during fetch setup; the consultation page is a reliable official pointer

9. **Guide to ICAAP**
   - Why:
     - improves supervisory capital-planning and management-action coverage
     - useful for questions tying governance, stress, and remediation together
   - Official source family:
     - https://www.bankingsupervision.europa.eu/press/pr/date/2018/html/ssm.pr181109.en.html
   - Note:
     - direct PDF should be resolved before manifest execution

10. **Guide to ILAAP**
    - Why:
      - complements liquidity and governance coverage
      - improves connections between intraday / liquidity supervision and management expectations
    - Official source family:
      - https://www.bankingsupervision.europa.eu/press/pr/date/2018/html/ssm.pr181109.en.html
    - Note:
      - direct PDF should be resolved before manifest execution

11. **Sound practices in counterparty credit risk governance and management**
    - Why:
      - useful governance / challenge / stress-testing / control-function evidence
      - helps broaden ECB-side governance material beyond RDARR
    - Official source:
      - https://www.bankingsupervision.europa.eu/press/pr/date/2023/html/ssm.pr231020.en.html

### PRA / Bank of England

12. **SS5/21 – The PRA’s approach to branch and subsidiary supervision**
    - Why:
      - directly strengthens cross-border and group-governance coverage
      - highly relevant to questions about wider-group information and supervisory visibility
    - Official source:
      - https://www.bankofengland.co.uk/prudential-regulation/publication/2021/july/pra-approach-to-branch-and-subsidiary-supervision-ss

13. **Internal governance of third-country branches**
    - Why:
      - strengthens branch governance and local implementation evidence
      - useful for governance-responsibility and control-ownership questions
    - Official source:
      - https://www.bankofengland.co.uk/prudential-regulation/publication/2016/internal-governance-of-third-country-branches-ss

14. **SS1/21 – Operational resilience: impact tolerances for important business services**
    - Why:
      - improves operational-resilience and service-continuity coverage
      - useful for risk-system resilience and control questions
    - Official source:
      - https://www.bankofengland.co.uk/prudential-regulation/publication/2021/march/operational-resilience-impact-tolerances-for-important-business-services-ss

15. **SS2/21 – Outsourcing and third-party risk management**
    - Why:
      - one of the highest-value PRA additions for outsourcing / material third-party / register / governance expectations
      - directly relevant to current finreg question families
    - Official source family:
      - https://www.bankofengland.co.uk/prudential-regulation/regulatory-digest/2021/march
    - Direct PDF:
      - https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2024/ss221-november-2024-update.pdf

16. **SoP1/21 – Operational resilience**
    - Why:
      - complements SS1/21 with the PRA's broader policy framing on governance, operational risk, continuity, and outsourcing
      - useful for cross-document operational-resilience comparisons
    - Official source:
      - https://www.bankofengland.co.uk/prudential-regulation/publication/2021/march/operational-resilience-sop

## Conditional Phase-2 Additions

These should be added only if detector/gating analysis still shows evidence gaps.

### Federal Reserve / OCC

17. **Federal Reserve SR 11-7: Guidance on Model Risk Management**
    - Why:
      - best conditional US addition for model-governance coverage
      - directly useful if model-risk questions remain weak
    - Official source:
      - https://www.federalreserve.gov/bankinforeg/srletters/sr1107.htm

18. **OCC Comptroller's Handbook: Model Risk Management**
    - Why:
      - strengthens the US-side model-governance comparator
      - useful if the project keeps any model-risk ambiguity slice
    - Official source:
      - https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/model-risk-management/index-model-risk-management.html

## Recommended Stop Condition

Stop corpus work when all three are true:

1. the corpus reaches roughly `30-50` documents
2. each active question family has at least two usable supervisory source families behind it
3. remaining failures are mostly retrieval / detector / gate issues rather than obvious evidence absence

At that point, corpus work is no longer the main project.

## Practical Next Step

Build a **phase-1.5 fetch manifest** from the priority additions above, fetch them,
rebuild the processed corpus, and rerun the refined `20Q` / `50Q` baselines once.

After that, freeze the corpus and move fully to:

- hallucination detector analysis
- stochastic gating analysis
- detector/gate ablations
