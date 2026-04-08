# FinReg Phase-1 Document Inventory

## Purpose

This document lists the first concrete candidate source documents for the real finreg corpus.

It is designed to support:

- the current finreg 20Q seed set
- the phase-1 corpus scope
- raw document collection under `data/raw/finreg/...`

## Selection Rules

Documents are preferred when they are:

- official supervisory, prudential, or regulatory publications
- clearly relevant to at least one current finreg question family
- useful as either an anchor source or a comparator source

## Phase-1 Core Documents

### BCBS / BIS

1. **Principles for effective risk data aggregation and risk reporting (BCBS 239)**
   - Role: anchor source for BCBS 239, data lineage, board accountability, proportionality, timelines
   - Themes:
     - `fq01`
     - `fq03`
     - `fq06`
     - `fq07`
     - `fq08`
     - `fq09`
   - URL:
     - https://www.bis.org/publ/bcbs239.htm

2. **Implementation of the Principles for effective risk data aggregation and risk reporting (BCBS 239 Principles)**
   - Role: implementation and current-practice evidence for BCBS 239
   - Themes:
     - `fq03`
     - `fq06`
     - `fq07`
     - `fq08`
     - `fq09`
   - URL:
     - https://www.bis.org/publ/bcbs_nl36.htm

3. **Monitoring tools for intraday liquidity management**
   - Role: anchor source for intraday liquidity monitoring
   - Themes:
     - `fq18`
   - URL:
     - https://www.bis.org/publ/bcbs248.htm

4. **Principles for the effective management and supervision of climate-related financial risks**
   - Role: anchor source for climate governance and climate-risk management expectations
   - Themes:
     - `fq12`
   - URL:
     - https://www.bis.org/bcbs/publ/d532.htm

5. **Principles for the sound management of third-party risk**
   - Role: anchor source for outsourcing / third-party controls
   - Themes:
     - `fq13`
   - URL:
     - https://www.bis.org/bcbs/publ/d605.htm

### EBA

6. **Supervisory Review and Evaluation Process (SREP) and Pillar 2**
   - Role: anchor entry point for supervisory review themes and linked guidelines
   - Themes:
     - `fq02`
     - `fq04`
   - URL:
     - https://www.eba.europa.eu/regulation-and-policy/supervisory-review-and-evaluation-process-srep-and-pillar-2

7. **Guidelines on ICT Risk Assessment under the SREP**
   - Role: comparator source for data controls, ICT risk, outsourced risk systems
   - Themes:
     - `fq02`
     - `fq13`
   - URL:
     - https://eba.europa.eu/activities/single-rulebook/regulatory-activities/supervisory-review-and-evaluation-process-srep-and-pillar-2/guidelines-ict-risk-assessment-under-srep

8. **Guidelines on outsourcing arrangements**
   - Role: comparator source for outsourcing governance and critical functions
   - Themes:
     - `fq13`
   - URL:
     - https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-revised-guidelines-outsourcing-arrangements

9. **Guidelines on the management of ESG risks**
   - Role: comparator source for climate risk integration and governance
   - Themes:
     - `fq12`
   - URL:
     - https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/sustainable-finance/guidelines-management-esg-risks

10. **Internal governance hub**
   - Role: comparator source for governance, accountability, control structures
   - Themes:
     - `fq02`
     - `fq07`
     - `fq19`
   - URL:
     - https://www.eba.europa.eu/regulation-and-policy/internal-governance

11. **Model validation hub**
   - Role: comparator source for internal model governance and validation themes
   - Themes:
     - `fq05`
     - `fq11`
   - URL:
     - https://www.eba.europa.eu/regulation-and-policy/model-validation

### PRA / Bank of England

12. **SS24/15 – The PRA’s approach to supervising liquidity and funding risks**
   - Role: local comparator for liquidity and intraday monitoring expectations
   - Themes:
     - `fq18`
   - URL:
     - https://www.bankofengland.co.uk/prudential-regulation/publication/2015/the-pras-approach-to-supervising-liquidity-and-funding-risks-ss

13. **CP6/22 – Model risk management principles for banks**
   - Role: local comparator for model risk management and validation depth
   - Themes:
     - `fq05`
     - `fq11`
   - URL:
     - https://www.bankofengland.co.uk/prudential-regulation/publication/2022/june/model-risk-management-principles-for-banks

14. **SS5/25 – Enhancing banks’ and insurers’ approaches to managing climate-related risks**
   - Role: local comparator for climate-risk management and proportional implementation
   - Themes:
     - `fq12`
   - URL:
     - https://www.bankofengland.co.uk/prudential-regulation/publication/2025/december/enhancing-banks-and-insurers-approaches-to-managing-climate-related-risks-ss

15. **Operational reporting / technical reporting guidance from PRA/BoE**
   - Role: local comparator for reporting controls, audit trail, materiality, operational expectations
   - Themes:
     - `fq10`
     - `fq14`
     - `fq15`
     - `fq17`
   - Status:
     - existing repo notes suggest BoE reporting materials are important
   - Initial candidate families:
     - occasional consultation paper / PRA110 material
     - MREL instructions and templates

### ECB

16. **Guide on effective risk data aggregation and risk reporting**
   - Role: strong regional comparator for BCBS 239 implementation and governance expectations
   - Themes:
     - `fq06`
     - `fq07`
     - `fq08`
     - `fq09`
   - URL:
     - https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guides240503_riskreporting.en.pdf

17. **Guide on climate-related and environmental risks**
   - Role: regional comparator for climate-risk governance and integration
   - Themes:
     - `fq12`
   - URL:
     - https://www.bankingsupervision.europa.eu/framework/legal-framework/public-consultations/html/climate-related_risks_faq.en.html
   - Note:
     - the FAQ page is used here as a stable official pointer to the guide family

18. **ECB guide to internal models**
   - Role: comparator for model governance and internal model expectations
   - Themes:
     - `fq05`
     - `fq11`
   - URL:
     - https://www.bankingsupervision.europa.eu/press/other-publications/publications/html/ssm.faq_guide_internal_models_2025~baf433d505.de.html

19. **ECB 2025 stress test programme / related supervisory material**
   - Role: comparator for stress-testing and supervisory challenge themes
   - Themes:
     - `fq04`
   - URL:
     - https://www.bankingsupervision.europa.eu/press/pr/date/2025/html/ssm.pr250120~6e75fde026.en.html

## Conditional Phase-2 Documents

### Federal Reserve / OCC

These are useful, but should not block phase 1.

1. Federal Reserve SR 11-7 / model risk management family
   - Themes:
     - `fq05`
     - `fq11`

2. OCC model risk / governance guidance
   - Themes:
     - `fq05`
     - `fq11`
     - possibly `fq16`

3. US guidance relevant to AI-assisted credit risk governance
   - Themes:
     - `fq16`

## Recommended Raw Folder Mapping

```text
data/raw/finreg/
  bcbs/
  eba/
  pra_boe/
  ecb/
  reporting/
```

Suggested placement:

- BCBS items 1-5 -> `data/raw/finreg/bcbs/`
- EBA items 6-11 -> `data/raw/finreg/eba/`
- PRA/BoE items 12-15 -> `data/raw/finreg/pra_boe/`
- ECB items 16-19 -> `data/raw/finreg/ecb/`

## Priority Order For Collection

If collection must start small, use this order:

1. BCBS 239
2. ECB RDARR guide
3. EBA SREP / ICT SREP materials
4. BCBS intraday liquidity tools
5. BCBS climate-risk principles
6. PRA model risk principles
7. EBA outsourcing guidelines
8. PRA / BoE operational reporting materials

## What This Inventory Is For

This inventory is enough to begin:

1. raw file collection
2. raw folder population
3. ingestion pipeline testing
4. coverage audit against the current finreg 20Q set

It is not yet the final canonical source list, but it is concrete enough to start corpus construction.
