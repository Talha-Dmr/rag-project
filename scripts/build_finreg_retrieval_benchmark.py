#!/usr/bin/env python3
"""
Build a corpus-grounded 50-question financial regulation retrieval benchmark.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "finreg"
OUTPUT_DIR = PROJECT_ROOT / "output" / "benchmarks"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_finreg_retrieval_50.json"
SUMMARY_PATH = OUTPUT_DIR / "benchmark_finreg_retrieval_50_summary.md"

ALLOWED_TYPES = {"definition", "purpose", "requirement", "scope", "process", "comparison"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def _entry(
    question: str,
    answer: str,
    doc_id: str,
    layer: str,
    gold_chunks: List[str],
    difficulty: str,
    question_type: str,
) -> Dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "doc_id": doc_id,
        "layer": layer,
        "gold_chunks": gold_chunks,
        "difficulty": difficulty,
        "question_type": question_type,
    }


def build_entries() -> List[Dict[str, object]]:
    return [
        _entry(
            "What is the purpose of the Basel leverage ratio framework?",
            "To restrict the build-up of leverage and reinforce risk-based requirements with a simple non-risk-based backstop.",
            "baselframework",
            "basel",
            ["baselframework_2226"],
            "easy",
            "purpose",
        ),
        _entry(
            "How is the Basel leverage ratio defined?",
            "It is Tier 1 capital divided by the exposure measure, expressed as a percentage.",
            "baselframework",
            "basel",
            ["baselframework_3702"],
            "easy",
            "definition",
        ),
        _entry(
            "What percentage of standardised-approach RWAs forms the Basel output floor?",
            "72.5% of the standardised-approach RWA base.",
            "baselframework",
            "basel",
            ["baselframework_0427"],
            "easy",
            "requirement",
        ),
        _entry(
            "When the Basel output floor is binding, which RWA amount must be used for compliance?",
            "The higher of nominated-approach RWAs and 72.5% of standardised-approach RWAs must be used.",
            "baselframework",
            "basel",
            ["baselframework_0426", "baselframework_0427"],
            "medium",
            "comparison",
        ),
        _entry(
            "For Basel large exposure purposes, on what value basis are cryptoasset exposures included when they create credit risk exposure?",
            "They are included according to their accounting value.",
            "baselframework",
            "basel",
            ["baselframework_0181"],
            "easy",
            "requirement",
        ),
        _entry(
            "What should supervisors do if transactions are not adequately captured in the leverage ratio exposure measure or may lead to destabilising deleveraging?",
            "They should carefully scrutinise the transactions and consider actions to address the concern.",
            "baselframework",
            "basel",
            ["baselframework_2236"],
            "medium",
            "requirement",
        ),
        _entry(
            "How does the Basel framework say global systemic importance should be measured for G-SIB purposes?",
            "By the impact a bank's failure would have on the global financial system and wider economy, rather than by the likelihood of failure.",
            "baselframework",
            "basel",
            ["baselframework_0037"],
            "medium",
            "definition",
        ),
        _entry(
            "Which categories receive equal weight in the Basel G-SIB methodology?",
            "Size, cross-jurisdictional activity, interconnectedness, substitutability or financial institution infrastructure, and complexity.",
            "baselframework",
            "basel",
            ["baselframework_0037"],
            "medium",
            "definition",
        ),
        _entry(
            "In the Basel minority-interest calculation, what CET1 threshold is used in the lower-of test for the subsidiary's surplus CET1?",
            "The minimum CET1 requirement plus the capital conservation buffer, i.e. 7.0% of consolidated RWA.",
            "baselframework",
            "basel",
            ["baselframework_0290", "baselframework_0291"],
            "hard",
            "requirement",
        ),
        _entry(
            "How should cryptoasset exposures generally be treated for Basel LCR and NSFR purposes?",
            "They should follow a treatment consistent with traditional exposures that have economically equivalent risks.",
            "baselframework",
            "basel",
            ["baselframework_0165"],
            "medium",
            "comparison",
        ),
        _entry(
            "When may securities financing transactions with a single counterparty be measured on a net basis in the NSFR?",
            "Only when the LEV30 netting conditions for securities financing transactions are met.",
            "baselframework",
            "basel",
            ["baselframework_2632"],
            "medium",
            "process",
        ),
        _entry(
            "Before recognising tokenised assets as collateral for credit risk mitigation, what must banks assess?",
            "They must assess whether the collateral meets the relevant eligibility requirements, including prompt liquidation and legal certainty.",
            "baselframework",
            "basel",
            ["baselframework_0118"],
            "medium",
            "requirement",
        ),
        _entry(
            "Through which two components should operational risk from cryptoasset activities generally be captured under the Basel standardised approach?",
            "Through the Business Indicator and the Internal Loss Multiplier.",
            "baselframework",
            "basel",
            ["baselframework_0164"],
            "easy",
            "definition",
        ),
        _entry(
            "What are a bank's nominated approaches under Basel RBC20?",
            "They are all approaches used for regulatory capital requirements other than approaches used solely for output floor calculations.",
            "baselframework",
            "basel",
            ["baselframework_0426"],
            "medium",
            "definition",
        ),
        _entry(
            "If floored RWAs are higher than pre-floor RWAs in the Basel output floor example, which amount is used for compliance?",
            "The floored RWAs are used.",
            "baselframework",
            "basel",
            ["baselframework_0436"],
            "easy",
            "comparison",
        ),
        _entry(
            "According to DORA, whose digital operational resilience is being strengthened?",
            "The financial sector's digital operational resilience is being strengthened.",
            "eurlex-dora-32022r2554",
            "eu_regulations",
            ["eurlex-dora-32022r2554_0001"],
            "easy",
            "purpose",
        ),
        _entry(
            "Under MiCA, what is an e-money token?",
            "A crypto-asset that seeks to stabilise its value by referencing only one official currency.",
            "eurlex-mica-32023r1114",
            "eu_regulations",
            ["eurlex-mica-32023r1114_0026"],
            "easy",
            "definition",
        ),
        _entry(
            "Under MiCA, what is an asset-referenced token?",
            "A crypto-asset that seeks to stabilise its value by referencing another value or right, or a combination of them, including one or several official currencies.",
            "eurlex-mica-32023r1114",
            "eu_regulations",
            ["eurlex-mica-32023r1114_0026"],
            "easy",
            "definition",
        ),
        _entry(
            "What information should a MiCA crypto-asset white paper contain?",
            "It should contain mandatory disclosures on the issuer or offeror, the project, the offer or admission to trading, attached rights and obligations, underlying technology, and related risks.",
            "eurlex-mica-32023r1114",
            "eu_regulations",
            ["eurlex-mica-32023r1114_0033"],
            "medium",
            "requirement",
        ),
        _entry(
            "What does PSD2 say payment initiation services provide to a payee?",
            "They provide comfort that the payment has been initiated so the payee can release goods or deliver the service without undue delay.",
            "eurlex-psd2-32015l2366",
            "eu_regulations",
            ["eurlex-psd2-32015l2366_0028"],
            "easy",
            "purpose",
        ),
        _entry(
            "What overall benefit do PSD2 account information services give the payment service user?",
            "They allow the user to have an overall view of their financial situation immediately at any given moment.",
            "eurlex-psd2-32015l2366",
            "eu_regulations",
            ["eurlex-psd2-32015l2366_0028"],
            "easy",
            "purpose",
        ),
        _entry(
            "Under PSD2, which body prepares draft regulatory technical standards on strong customer authentication?",
            "The EBA prepares the draft regulatory technical standards, which the Commission is empowered to adopt.",
            "eurlex-psd2-32015l2366",
            "eu_regulations",
            ["eurlex-psd2-32015l2366_0114"],
            "medium",
            "process",
        ),
        _entry(
            "For payment initiation and account information services under PSD2, which data are expressly not treated as sensitive payment data?",
            "The account owner's name and the account number are not treated as sensitive payment data.",
            "eurlex-psd2-32015l2366",
            "eu_regulations",
            ["eurlex-psd2-32015l2366_0136"],
            "easy",
            "definition",
        ),
        _entry(
            "Which tools does BRRD list as resolution tools?",
            "The sale of business or shares, a bridge institution, asset separation, and bail-in.",
            "eurlex-brrd-32014l0059",
            "eu_regulations",
            ["eurlex-brrd-32014l0059_0057"],
            "medium",
            "definition",
        ),
        _entry(
            "What is the main purpose of a BRRD bridge institution?",
            "To ensure essential financial services continue to be provided and essential financial activities continue to be performed.",
            "eurlex-brrd-32014l0059",
            "eu_regulations",
            ["eurlex-brrd-32014l0059_0061"],
            "easy",
            "purpose",
        ),
        _entry(
            "Why does BRRD say general corporate insolvency procedures may be inappropriate for institutions?",
            "Because they may fail to ensure sufficient speed of intervention, continuation of critical functions, and preservation of financial stability.",
            "eurlex-brrd-32014l0059",
            "eu_regulations",
            ["eurlex-brrd-32014l0059_0004", "eurlex-brrd-32014l0059_0005"],
            "hard",
            "purpose",
        ),
        _entry(
            "In proven low-risk circumstances under AMLD4, which obligations may Member States relax for e-money products?",
            "They may exempt those products from certain customer due diligence measures such as identifying and verifying the customer and beneficial owner, but not from transaction or relationship monitoring.",
            "eurlex-amld4-32015l0849",
            "eu_regulations",
            ["eurlex-amld4-32015l0849_0008"],
            "hard",
            "comparison",
        ),
        _entry(
            "What measures does AMLD4 call for in relation to politically exposed persons?",
            "Appropriate enhanced customer due diligence measures.",
            "eurlex-amld4-32015l0849",
            "eu_regulations",
            ["eurlex-amld4-32015l0849_0031"],
            "easy",
            "requirement",
        ),
        _entry(
            "What are the three core objectives of IOSCO's securities regulation principles?",
            "Protecting investors, ensuring markets are fair efficient and transparent, and reducing systemic risk.",
            "ioscopd154",
            "iosco",
            ["ioscopd154_0001"],
            "easy",
            "definition",
        ),
        _entry(
            "What does IOSCO Principle 2 require of the regulator?",
            "The regulator should be operationally independent and accountable in the exercise of its functions and powers.",
            "ioscopd154",
            "iosco",
            ["ioscopd154_0001"],
            "easy",
            "requirement",
        ),
        _entry(
            "Under IOSCO Principle 8, what powers should the regulator have?",
            "Comprehensive inspection, investigation and surveillance powers.",
            "ioscopd154",
            "iosco",
            ["ioscopd154_0002"],
            "easy",
            "requirement",
        ),
        _entry(
            "According to the 2017 IOSCO Principles, what systemic-risk process should the regulator have or contribute to?",
            "A process to identify, monitor, mitigate and manage systemic risk appropriate to its mandate.",
            "ioscopd561",
            "iosco",
            ["ioscopd561_0003"],
            "medium",
            "process",
        ),
        _entry(
            "What should regulation be designed to detect and deter under the IOSCO 2017 principles?",
            "Manipulation and other unfair trading practices.",
            "ioscopd561",
            "iosco",
            ["ioscopd561_0008"],
            "easy",
            "requirement",
        ),
        _entry(
            "Under IOSCO Outsourcing Principle 1, what should a regulated entity do when selecting a service provider?",
            "It should conduct suitable due diligence and monitor the provider's ongoing performance.",
            "ioscopd687",
            "iosco",
            ["ioscopd687_0058"],
            "easy",
            "process",
        ),
        _entry(
            "Under IOSCO Outsourcing Principle 2, what kind of contract is required with each service provider?",
            "A legally binding written contract whose detail matches the materiality or criticality of the outsourced task.",
            "ioscopd687",
            "iosco",
            ["ioscopd687_0069"],
            "easy",
            "requirement",
        ),
        _entry(
            "Under IOSCO Outsourcing Principle 3, what must procedures and controls protect and ensure?",
            "They must protect proprietary and client-related information and software, and ensure continuity of service with disaster recovery and tested backups.",
            "ioscopd687",
            "iosco",
            ["ioscopd687_0004"],
            "medium",
            "requirement",
        ),
        _entry(
            "According to IOSCO's 2022 commodity derivatives principles, what fundamental functions should commodity derivatives markets serve?",
            "They should serve price discovery and hedging functions while operating free from manipulation and abusive trading schemes.",
            "ioscopd726",
            "iosco",
            ["ioscopd726_0042"],
            "easy",
            "purpose",
        ),
        _entry(
            "What information must relevant market authorities obtain to apply position limits and carry out position management in commodity derivatives markets?",
            "They need position information by holder and contract month, including data that identifies common ownership and control.",
            "ioscopd726",
            "iosco",
            ["ioscopd726_0162"],
            "hard",
            "requirement",
        ),
        _entry(
            "Which mandatory entity-level fields must be completed for each reported entity in the BoE MREL reporting templates?",
            "Entity name, FRN, LEI, basis of reporting, reporting period start and end dates, reporting currency, and working-contact name, position and email.",
            "instructions-and-templates",
            "technical_reporting",
            ["instructions-and-templates_0001"],
            "hard",
            "requirement",
        ),
        _entry(
            "How should amounts be entered in the BoE MREL reporting templates?",
            "They should be reported in absolute full amounts, not abbreviated values such as thousands.",
            "instructions-and-templates",
            "technical_reporting",
            ["instructions-and-templates_0001"],
            "easy",
            "requirement",
        ),
        _entry(
            "What three MREL reporting templates are listed in the BoE guidance?",
            "MRL001 MREL Resources, MRL002 MREL Resources Forecast, and MRL003 MREL Debt.",
            "instructions-and-templates",
            "technical_reporting",
            ["instructions-and-templates_0002"],
            "medium",
            "definition",
        ),
        _entry(
            "What does PRA110 aim to capture?",
            "The maturity mismatch of an institution's activities.",
            "occasional-consultation-paper-march-2020",
            "technical_reporting",
            ["occasional-consultation-paper-march-2020_0001"],
            "easy",
            "purpose",
        ),
        _entry(
            "In PRA110, what should the 'Outflows and inflows' section cover?",
            "Future contractual cash flows from all on- and off-balance-sheet items, based on contracts valid at the reporting date.",
            "occasional-consultation-paper-march-2020",
            "technical_reporting",
            ["occasional-consultation-paper-march-2020_0002"],
            "medium",
            "scope",
        ),
        _entry(
            "How must contractual flows be allocated across PRA110 buckets?",
            "Across 108 time buckets according to residual maturity, with days counted as calendar days.",
            "occasional-consultation-paper-march-2020",
            "technical_reporting",
            ["occasional-consultation-paper-march-2020_0003"],
            "easy",
            "process",
        ),
        _entry(
            "What is a 'material third-party arrangement' in the PRA operational incident and third-party reporting instrument?",
            "It is an arrangement so important that disruption could endanger safety and soundness, policyholder protection, or UK financial stability for certain firms.",
            "operational-incident-and-third-party-reporting-policy-statement",
            "technical_reporting",
            ["operational-incident-and-third-party-reporting-policy-statement_0003"],
            "medium",
            "definition",
        ),
        _entry(
            "Why did the EBA issue guidelines on restrictive-measures compliance?",
            "To set common EU standards for governance and controls because divergent supervisory expectations were making compliance less effective and more risky.",
            "eba_principles",
            "supporting_explanatory",
            ["eba_principles_0002", "eba_principles_0003"],
            "hard",
            "purpose",
        ),
        _entry(
            "According to the EBA report, what procedures did the European Commission highlight as necessary to avoid breaches of restrictive-measures regulations?",
            "Screening, risk assessment, multi-level due diligence, and ongoing monitoring.",
            "eba_principles",
            "supporting_explanatory",
            ["eba_principles_0007"],
            "medium",
            "process",
        ),
        _entry(
            "In the PRA climate-risk consultation, what is Step 1 of the proportionality process?",
            "Step 1 is to identify material climate-related risks and understand their effect on business-model resilience across time horizons and climate scenarios, supported by scenario analysis.",
            "enhancing-banks-and-insurers-approaches-to-managing-climate-related-risks-consultation-paper",
            "supporting_explanatory",
            ["enhancing-banks-and-insurers-approaches-to-managing-climate-related-risks-consultation-paper_0018"],
            "hard",
            "process",
        ),
        _entry(
            "How does the PRA consultation define funded reinsurance?",
            "As a collateralised quota share reinsurance contract that transfers part or all of the asset and liability risks of an annuity portfolio to a counterparty.",
            "funded-reinsurance-consultation-paper",
            "supporting_explanatory",
            ["funded-reinsurance-consultation-paper_0003"],
            "easy",
            "definition",
        ),
        _entry(
            "Which firms does the PRA's international-banks supervisory statement apply to?",
            "PRA-authorised banks and designated investment firms headquartered outside the UK or in groups based outside the UK, operating in the UK as branches or subsidiaries.",
            "international-firms-upates-to-ss521-policy-statement",
            "supporting_explanatory",
            ["international-firms-upates-to-ss521-policy-statement_0002"],
            "medium",
            "scope",
        ),
    ]


def load_chunk_registry() -> Dict[str, str]:
    registry: Dict[str, str] = {}
    for path in PROCESSED_ROOT.rglob("*.chunks.jsonl"):
        doc_id = path.name.replace(".chunks.jsonl", "")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                registry[record["chunk_id"]] = doc_id
    return registry


def validate_entries(entries: List[Dict[str, object]], registry: Dict[str, str]) -> None:
    if len(entries) != 50:
        raise SystemExit(f"Expected 50 entries, found {len(entries)}")

    for index, entry in enumerate(entries, start=1):
        difficulty = str(entry["difficulty"])
        question_type = str(entry["question_type"])
        gold_chunks = list(entry["gold_chunks"])
        doc_id = str(entry["doc_id"])

        if difficulty not in ALLOWED_DIFFICULTIES:
            raise SystemExit(f"Invalid difficulty for q_{index:03d}: {difficulty}")
        if question_type not in ALLOWED_TYPES:
            raise SystemExit(f"Invalid question_type for q_{index:03d}: {question_type}")
        if not 1 <= len(gold_chunks) <= 3:
            raise SystemExit(f"Invalid gold chunk count for q_{index:03d}")
        if not str(entry["question"]).strip():
            raise SystemExit(f"Empty question for q_{index:03d}")
        if not str(entry["answer"]).strip():
            raise SystemExit(f"Empty answer for q_{index:03d}")

        for chunk_id in gold_chunks:
            owner = registry.get(chunk_id)
            if owner is None:
                raise SystemExit(f"Missing chunk id {chunk_id} for q_{index:03d}")
            if owner != doc_id:
                raise SystemExit(
                    f"Chunk/doc mismatch for q_{index:03d}: chunk {chunk_id} belongs to {owner}, not {doc_id}"
                )


def attach_ids(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for index, entry in enumerate(entries, start=1):
        item = dict(entry)
        item["id"] = f"q_{index:03d}"
        output.append(item)
    return output


def write_summary(entries: List[Dict[str, object]]) -> None:
    layer_counts = Counter(str(entry["layer"]) for entry in entries)
    difficulty_counts = Counter(str(entry["difficulty"]) for entry in entries)
    question_type_counts = Counter(str(entry["question_type"]) for entry in entries)

    lines = [
        "# FinReg Retrieval Benchmark Summary",
        "",
        f"- Total questions: {len(entries)}",
        "",
        "## Questions per Category",
        "",
    ]
    for key in sorted(layer_counts):
        lines.append(f"- {key}: {layer_counts[key]}")

    lines.extend(
        [
            "",
            "## Questions per Difficulty",
            "",
        ]
    )
    for key in ("easy", "medium", "hard"):
        lines.append(f"- {key}: {difficulty_counts.get(key, 0)}")

    lines.extend(
        [
            "",
            "## Questions per Question Type",
            "",
        ]
    )
    for key in sorted(question_type_counts):
        lines.append(f"- {key}: {question_type_counts[key]}")

    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- The processed corpus did not contain populated `academic`, `technical_manuals`, or `supporting_docs` layer-4 folders, so the lower-priority buckets were approximated with BoE and EBA explanatory or implementation-oriented materials.",
            "- US regulations were not sampled in the benchmark because the requested coverage plan explicitly prioritised Basel, EU regulation, IOSCO, technical or reporting materials, and supporting explanatory documents.",
            "- Gold evidence was restricted to local chunk support only; hard items use at most 2 nearby chunks.",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_chunk_registry()
    entries = build_entries()
    validate_entries(entries, registry)
    benchmark = attach_ids(entries)

    BENCHMARK_PATH.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(benchmark)

    print(f"Wrote {BENCHMARK_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
