#!/usr/bin/env python3
"""Build a small FinReg-aware NLI dataset for detector adaptation."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


GROUPS: list[dict[str, Any]] = [
    {
        "source_org": "BCBS",
        "theme": "rdarr",
        "source_file": "data/processed/finreg/bcbs/bcbs239_impl_note.txt",
        "premise": "Accurate, comprehensive and timely data aggregation and reporting capabilities are critical for identifying and managing material risks that could result in financial losses.",
        "entailment": "Risk data aggregation and reporting capabilities help banks identify and manage material risks.",
        "contradiction": "Risk data aggregation and reporting capabilities are not relevant to identifying material risks.",
        "neutral": "PRA outsourcing guidance includes expectations for written outsourcing agreements.",
    },
    {
        "source_org": "BCBS",
        "theme": "rdarr",
        "source_file": "data/processed/finreg/bcbs/bcbs239_impl_note.txt",
        "premise": "Bank boards have the responsibility for broad oversight of risk data aggregation activities and must gain assurance from management that processes are sound.",
        "entailment": "Boards have oversight responsibility for risk data aggregation activities.",
        "contradiction": "Boards have no oversight responsibility for risk data aggregation activities.",
        "neutral": "The ECB climate guide discusses physical and transition risk drivers.",
    },
    {
        "source_org": "BCBS",
        "theme": "rdarr",
        "source_file": "data/processed/finreg/bcbs/bcbs239_impl_note.txt",
        "premise": "Data lineage, or the traceability of data from its origin to its final use, is important for confirming data quality.",
        "entailment": "Data lineage helps confirm data quality by tracing data from origin to final use.",
        "contradiction": "Data lineage is unrelated to confirming data quality.",
        "neutral": "SR 11-7 says model validation should include ongoing monitoring.",
    },
    {
        "source_org": "BCBS",
        "theme": "rdarr",
        "source_file": "data/processed/finreg/bcbs/bcbs239_impl_note.txt",
        "premise": "The ability to produce timely, accurate and complete ad-hoc reports remains a significant hurdle for some banks, particularly during crises or in response to regulatory requests.",
        "entailment": "Some banks still struggle to produce timely, accurate and complete ad-hoc reports.",
        "contradiction": "Banks have no significant difficulty producing timely, accurate and complete ad-hoc reports during crises.",
        "neutral": "EBA remuneration guidance addresses sound remuneration policies.",
    },
    {
        "source_org": "BCBS",
        "theme": "rdarr",
        "source_file": "data/processed/finreg/bcbs/bcbs239_impl_note.txt",
        "premise": "Internationally active banks face particular challenges in aligning data management practices across subsidiaries and local affiliates in various countries.",
        "entailment": "Cross-border group structures can complicate alignment of data management practices.",
        "contradiction": "International activity eliminates challenges in aligning data management practices across subsidiaries.",
        "neutral": "The PRA climate statement covers governance, risk management, scenario analysis, data and disclosures.",
    },
    {
        "source_org": "BCBS",
        "theme": "liquidity",
        "source_file": "data/processed/finreg/bcbs/bcbs248_intraday_liquidity.txt",
        "premise": "The monitoring tools were developed to enable banking supervisors to better monitor a bank's management of intraday liquidity risk and its ability to meet payment and settlement obligations on a timely basis.",
        "entailment": "BCBS intraday liquidity monitoring tools help supervisors monitor payment and settlement obligations.",
        "contradiction": "BCBS intraday liquidity tools are unrelated to payment and settlement obligations.",
        "neutral": "BCBS third-party principles address risks from service provider dependencies.",
    },
    {
        "source_org": "BCBS",
        "theme": "climate",
        "source_file": "data/processed/finreg/bcbs/bcbs_climate_principles.txt",
        "premise": "The Basel Committee published principles for the effective management and supervision of climate-related financial risks to the global banking system.",
        "entailment": "BCBS climate principles concern management and supervision of climate-related financial risks.",
        "contradiction": "BCBS climate principles say climate-related financial risks are outside banking supervision.",
        "neutral": "EBA outsourcing guidelines distinguish outsourcing from other third-party arrangements.",
    },
    {
        "source_org": "BCBS",
        "theme": "third_party",
        "source_file": "data/processed/finreg/bcbs/bcbs_third_party_risk.txt",
        "premise": "Banks' increased dependency on third-party service providers necessitates evolving the traditional concept of outsourcing to encompass a broader range of third-party arrangements.",
        "entailment": "BCBS third-party risk principles recognize that outsourcing concepts must cover broader third-party arrangements.",
        "contradiction": "BCBS says increased dependency on third-party service providers does not affect outsourcing concepts.",
        "neutral": "ECB stress testing can feed qualitative findings into SREP assessments.",
    },
    {
        "source_org": "EBA",
        "theme": "governance",
        "source_file": "data/processed/finreg/eba/eba_internal_governance.txt",
        "premise": "Sound internal governance practices helped some institutions manage the financial crisis better than others and support effective risk management.",
        "entailment": "EBA internal governance guidance links sound governance practices with better risk management.",
        "contradiction": "EBA internal governance guidance says sound governance practices weaken risk management.",
        "neutral": "BCBS intraday liquidity monitoring reporting commenced monthly from 1 January 2015.",
    },
    {
        "source_org": "EBA",
        "theme": "governance",
        "source_file": "data/processed/finreg/eba/eba_internal_governance.txt",
        "premise": "Internal governance frameworks include internal control mechanisms and risk management arrangements.",
        "entailment": "Internal control mechanisms are part of internal governance frameworks.",
        "contradiction": "Internal control mechanisms are excluded from internal governance frameworks.",
        "neutral": "SR 11-7 discusses model inventories for models implemented, under development, or retired.",
    },
    {
        "source_org": "EBA",
        "theme": "outsourcing",
        "source_file": "data/processed/finreg/eba/eba_outsourcing.txt",
        "premise": "Outsourcing arrangements should not lead to a delegation by senior management of its responsibility and accountability.",
        "entailment": "EBA outsourcing guidance does not let senior management delegate its accountability through outsourcing.",
        "contradiction": "EBA outsourcing guidance allows senior management to transfer accountability to the service provider.",
        "neutral": "The ECB climate guide explains physical and transition risks.",
    },
    {
        "source_org": "EBA",
        "theme": "ict",
        "source_file": "data/processed/finreg/eba/eba_ict_security_risk_management.txt",
        "premise": "Institutions should identify, establish and maintain updated mapping of their business functions, roles and supporting processes to identify the importance of each and their interdependencies related to ICT risk.",
        "entailment": "EBA ICT guidance expects institutions to map business functions and dependencies related to ICT risk.",
        "contradiction": "EBA ICT guidance says institutions should avoid mapping business functions related to ICT risk.",
        "neutral": "BCBS 239 focuses on risk data aggregation and risk reporting.",
    },
    {
        "source_org": "ECB",
        "theme": "climate",
        "source_file": "data/processed/finreg/ecb/ecb_climate_guide_family.txt",
        "premise": "Physical risks refer to financial impacts of a changing climate, more frequent extreme weather events, gradual climate changes and environmental degradation.",
        "entailment": "The ECB climate guide treats physical risks as financial impacts from climate and environmental changes.",
        "contradiction": "The ECB climate guide says physical risks have no financial impact.",
        "neutral": "PRA third-country branch guidance concerns branch and subsidiary supervision.",
    },
    {
        "source_org": "ECB",
        "theme": "climate",
        "source_file": "data/processed/finreg/ecb/ecb_climate_guide_family.txt",
        "premise": "Transition risks can arise from the process of adjustment towards a lower-carbon and more sustainable economy, including policy changes, technological progress or changes in market sentiment.",
        "entailment": "Transition risks may come from policy, technology or market sentiment changes during the move to a lower-carbon economy.",
        "contradiction": "Transition risks cannot arise from policy changes or technology changes.",
        "neutral": "Fed SR 11-7 says model risk can arise from incorrect or misused model outputs.",
    },
    {
        "source_org": "ECB",
        "theme": "climate",
        "source_file": "data/processed/finreg/ecb/ecb_climate_guide_family.txt",
        "premise": "Most banks have yet to develop a comprehensive and forward-looking risk management approach for climate-related and environmental risks.",
        "entailment": "The ECB found many banks still lacked comprehensive forward-looking climate risk management.",
        "contradiction": "The ECB found most banks already had comprehensive forward-looking climate risk management.",
        "neutral": "EBA internal governance guidance discusses the management body in management and supervisory functions.",
    },
    {
        "source_org": "ECB",
        "theme": "stress_testing",
        "source_file": "data/processed/finreg/ecb/ecb_2025_stress_test.txt",
        "premise": "The 2025 stress test will focus on assessing the quality of the data provided by banks, given the importance of risk data aggregation and reporting capabilities for banks' resilience.",
        "entailment": "The ECB 2025 stress test focuses partly on data quality and risk data aggregation capabilities.",
        "contradiction": "The ECB 2025 stress test ignores the quality of data provided by banks.",
        "neutral": "BCBS third-party principles set a common baseline for banks and supervisors.",
    },
    {
        "source_org": "ECB",
        "theme": "srep",
        "source_file": "data/processed/finreg/ecb/ecb_icaap_ilaap_consultation.txt",
        "premise": "Joint Supervisory Teams take ICAAP and ILAAP information packages into account in annual assessments conducted as part of the Supervisory Review and Evaluation Process.",
        "entailment": "ICAAP and ILAAP submissions feed into JST annual SREP assessments.",
        "contradiction": "ICAAP and ILAAP information packages are not considered in SREP assessments.",
        "neutral": "PRA outsourcing guidance covers sub-outsourcing arrangements.",
    },
    {
        "source_org": "ECB",
        "theme": "model_risk",
        "source_file": "data/processed/finreg/ecb/ecb_internal_models_guide.txt",
        "premise": "The chapter on overarching principles for internal models covers non-model-specific principles applicable to all risk types, including model risk management framework, data governance and use of machine learning techniques.",
        "entailment": "ECB internal model guidance includes model risk management, data governance and machine learning principles.",
        "contradiction": "ECB internal model guidance excludes model risk management and data governance from overarching principles.",
        "neutral": "BCBS 239 was finalized after consultation in June 2012.",
    },
    {
        "source_org": "PRA-BoE",
        "theme": "outsourcing",
        "source_file": "data/processed/finreg/pra_boe/pra_ss2_21_outsourcing_third_party.txt",
        "premise": "The supervisory statement clarifies how the PRA expects banks to approach the EBA outsourcing guidelines in the context of PRA requirements and expectations.",
        "entailment": "PRA outsourcing guidance clarifies how banks should approach EBA outsourcing guidelines under PRA expectations.",
        "contradiction": "PRA outsourcing guidance says EBA outsourcing guidelines are irrelevant to banks.",
        "neutral": "ECB climate guidance is not legally binding.",
    },
    {
        "source_org": "PRA-BoE",
        "theme": "outsourcing",
        "source_file": "data/processed/finreg/pra_boe/pra_ss2_21_outsourcing_third_party.txt",
        "premise": "Chapter 5 sets out PRA expectations for firms during the pre-outsourcing phase and addresses materiality and risk assessments of outsourcing and other third-party arrangements.",
        "entailment": "PRA expects firms to assess materiality and risk before outsourcing.",
        "contradiction": "PRA says firms do not need materiality or risk assessments before outsourcing.",
        "neutral": "BCBS climate principles address financial risks in the global banking system.",
    },
    {
        "source_org": "PRA-BoE",
        "theme": "operational_resilience",
        "source_file": "data/processed/finreg/pra_boe/pra_ss1_21_operational_resilience.txt",
        "premise": "Operational resilience policy requires firms to identify important business services and set impact tolerances for each important business service.",
        "entailment": "PRA operational resilience expectations include identifying important business services and setting impact tolerances.",
        "contradiction": "PRA operational resilience expectations say firms should not set impact tolerances.",
        "neutral": "SR 11-7 describes effective challenge as objective critical analysis.",
    },
    {
        "source_org": "PRA-BoE",
        "theme": "climate",
        "source_file": "data/processed/finreg/pra_boe/pra_ss5_25_climate.txt",
        "premise": "The approach is proportionate, practical and reflects the evolving climate-related risk landscape.",
        "entailment": "PRA climate expectations are intended to be proportionate and practical.",
        "contradiction": "PRA climate expectations are intended to be disproportionate and impractical.",
        "neutral": "BCBS 239 discusses ad-hoc risk reports during stress.",
    },
    {
        "source_org": "PRA-BoE",
        "theme": "branch_governance",
        "source_file": "data/processed/finreg/pra_boe/pra_ss5_21_branch_subsidiary.txt",
        "premise": "The PRA's expectations are intended to help firms understand the PRA's approach to supervising international banks operating in the UK through branches and subsidiaries.",
        "entailment": "PRA branch and subsidiary guidance concerns supervision of international banks operating in the UK.",
        "contradiction": "PRA branch and subsidiary guidance is unrelated to international banks operating in the UK.",
        "neutral": "EBA ICT guidance expects institutions to maintain updated ICT risk mappings.",
    },
    {
        "source_org": "Federal Reserve",
        "theme": "model_risk",
        "source_file": "data/processed/finreg/fed_occ/fed_sr11_7_model_risk.txt",
        "premise": "Model risk can lead to financial loss, poor business and strategic decision-making, or damage to a banking organization's reputation.",
        "entailment": "SR 11-7 says model risk can cause financial, strategic and reputational harm.",
        "contradiction": "SR 11-7 says model risk cannot cause financial loss or reputational harm.",
        "neutral": "ECB ICAAP and ILAAP packages are submitted to Joint Supervisory Teams.",
    },
    {
        "source_org": "Federal Reserve",
        "theme": "model_risk",
        "source_file": "data/processed/finreg/fed_occ/fed_sr11_7_model_risk.txt",
        "premise": "Model risk increases with greater model complexity, higher uncertainty about inputs and assumptions, broader extent of use, and larger potential impact.",
        "entailment": "SR 11-7 links higher model risk to greater complexity, uncertainty, use and impact.",
        "contradiction": "SR 11-7 says model risk decreases as model complexity and uncertainty increase.",
        "neutral": "PRA operational resilience policy includes important business services.",
    },
    {
        "source_org": "Federal Reserve",
        "theme": "model_risk",
        "source_file": "data/processed/finreg/fed_occ/fed_sr11_7_model_risk.txt",
        "premise": "A sound development process includes a clear statement of purpose, sound design, robust methodologies, rigorous data quality assessment and appropriate documentation.",
        "entailment": "SR 11-7 expects model development to include purpose, design, methodology, data quality assessment and documentation.",
        "contradiction": "SR 11-7 says model development does not need documentation or data quality assessment.",
        "neutral": "BCBS intraday liquidity tools complement qualitative liquidity guidance.",
    },
    {
        "source_org": "Federal Reserve",
        "theme": "model_risk",
        "source_file": "data/processed/finreg/fed_occ/fed_sr11_7_model_risk.txt",
        "premise": "Validation activities should continue on an ongoing basis after a model goes into use to track known model limitations and identify any new ones.",
        "entailment": "SR 11-7 expects model validation to continue after model implementation.",
        "contradiction": "SR 11-7 says validation should stop once a model goes into use.",
        "neutral": "EBA outsourcing guidance addresses sub-outsourcing and access rights.",
    },
    {
        "source_org": "Federal Reserve",
        "theme": "model_risk",
        "source_file": "data/processed/finreg/fed_occ/fed_sr11_7_model_risk.txt",
        "premise": "Strong governance includes documentation of model development and validation that is sufficiently detailed to allow parties unfamiliar with a model to understand how it operates.",
        "entailment": "SR 11-7 expects detailed model documentation so unfamiliar parties can understand model operation.",
        "contradiction": "SR 11-7 says model documentation should be too sparse for unfamiliar parties to understand the model.",
        "neutral": "ECB 2025 stress testing includes a counterparty credit risk exploratory scenario.",
    },
]


def make_examples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, group in enumerate(GROUPS, start=1):
        base_meta = {
            "source_org": group["source_org"],
            "theme": group["theme"],
            "source_file": group["source_file"],
            "builder": "build_finreg_detector_nli_dataset.py",
            "group_id": f"fdg{idx:03d}",
        }
        for label in ("entailment", "contradiction", "neutral"):
            rows.append(
                {
                    "id": f"fdg{idx:03d}_{label}",
                    "premise": group["premise"],
                    "hypothesis": group[label],
                    "label": label,
                    "metadata": {
                        **base_meta,
                        "pair_type": label,
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], numeric_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = dict(row)
            if numeric_labels:
                out["label"] = LABEL_TO_ID[out["label"]]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "source_orgs": dict(Counter(row["metadata"]["source_org"] for row in rows)),
        "themes": dict(Counter(row["metadata"]["theme"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/training/nli_dataset_finreg_detector_v1")
    parser.add_argument("--test-source", default="data/domain_finreg/detector_eval_finreg_v1.jsonl")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-groups", type=int, default=6)
    parser.add_argument(
        "--numeric-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write labels as 0/1/2. The training loader also accepts string labels.",
    )
    args = parser.parse_args()

    rows = make_examples()
    group_ids = sorted({row["metadata"]["group_id"] for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)

    val_group_ids = set(group_ids[: args.val_groups])
    train_rows = [row for row in rows if row["metadata"]["group_id"] not in val_group_ids]
    val_rows = [row for row in rows if row["metadata"]["group_id"] in val_group_ids]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows, args.numeric_labels)
    write_jsonl(output_dir / "val.jsonl", val_rows, args.numeric_labels)

    test_source = Path(args.test_source)
    if test_source.exists():
        shutil.copyfile(test_source, output_dir / "test.jsonl")
    else:
        raise FileNotFoundError(test_source)

    summary = {
        "seed": args.seed,
        "val_groups": args.val_groups,
        "numeric_labels": args.numeric_labels,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test_source": str(test_source),
    }
    test_rows = []
    with (output_dir / "test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))
    summary["test"] = summarize(test_rows)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
