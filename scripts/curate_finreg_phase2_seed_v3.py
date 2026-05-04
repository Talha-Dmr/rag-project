#!/usr/bin/env python3
"""
Extend the Phase 2 seed-v2 review CSV with a second expansion focused on
supported controls and clear contradicted comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


REVIEW_FIELDS = [
    "id",
    "review_priority",
    "question",
    "suggested_label",
    "suggested_error_type",
    "likely_label",
    "likely_error_type",
    "reviewer_confidence",
    "review_note_draft",
    "gold_label",
    "gold_error_type",
    "review_notes",
]


SECOND_EXPANSION: Dict[str, Dict[str, str]] = {
    "targeted_v2::fq03": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Clean sanity/control answer grounded on traceability, auditability, and remediation.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "supported control: correctly ties data lineage to traceability, auditability, and faster remediation",
    },
    "targeted_v2::fq04": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Direct stress-testing definition with no visible contamination or comparative overreach.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "supported control: stress testing is correctly described as evaluating capital adequacy and risk concentration under adverse scenarios",
    },
    "targeted_v2::fq10": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer is concise and grounded in the retrieved climate-risk governance statement.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "supported control: climate risk is treated as a governance priority because it must be reflected in governance, measurement, and planning",
    },
    "targeted_v2::fq35": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Comparative answer matches the retrieved BCBS vs EBA explainability emphases well enough for a supported comparison control.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "supported comparison: BCBS emphasizes decision-relevant explainability and challenge, while EBA emphasizes prudential and stakeholder transparency concerns",
    },
    "fever_local::fq40": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Despite boilerplate, the core comparison between BCBS integration and ECB acceleration is grounded.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "supported comparison: BCBS frames climate risk as integrated governance/measurement/planning, whereas ECB emphasizes faster supervisory timelines",
    },
    "fever_local::fq12": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "high",
        "review_note_draft": "Response assigns delegated control ownership to BCBS even though the retrieved BCBS chunk says explicit board-level accountability.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "contradicted by source: delegated control ownership belongs to EBA-style guidance, while BCBS retrieved evidence states direct board/senior-management accountability",
    },
    "fever_local::fq37": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "high",
        "review_note_draft": "Answer says ECB allows weak control evidence and incomplete audit trails, which reverses the retrieved ECB condition requiring robust controls and traceability.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "direct contradiction: ECB evidence conditions manual workarounds on robust control evidence and audit trails, not weak/incomplete ones",
    },
    "fever_local::fq47": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "medium",
        "review_note_draft": "The answer calls BCBS and Fed aligned by focusing on generic uncertainty language, but the retrieved stances differ on explicit ownership vs governance-evidence emphasis.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "false alignment claim: BCBS emphasizes direct accountability, while Fed emphasizes governance effectiveness over formal ownership language",
    },
    "targeted_v2::fq47": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "medium",
        "review_note_draft": "This is another explicit alignment claim that collapses a real BCBS vs Fed difference into generic caution language.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "contradicted comparison: response says thresholds align, but retrieved guidance contrasts explicit board ownership with governance-evidence supervision",
    },
    "targeted_v2::fq50": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer misidentifies the Federal Reserve as EBA/ECB-style climate guidance and then builds the comparison on the wrong institution.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "wrong-institution contradiction: response answers a BCBS vs Federal Reserve question using EBA/ECB climate-integration claims",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate FinReg Phase 2 seed-v3 review CSV")
    parser.add_argument(
        "--base-review-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_seed_v2.csv",
        help="Existing seed-v2 reviewed CSV to extend",
    )
    parser.add_argument(
        "--benchmark-prefill",
        default="evaluation_results/finreg_detector_phase2_prefill_smoke/benchmark_prefill.jsonl",
        help="Benchmark prefill used as question fallback",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_seed_v3.csv",
        help="Output merged review CSV path",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_question_map(benchmark_prefill: Path) -> Dict[str, str]:
    questions: Dict[str, str] = {}
    for row in load_jsonl(benchmark_prefill):
        provenance = row.get("provenance") or {}
        source_id = provenance.get("source_id", "")
        if source_id:
            questions[source_id] = row.get("question", "")

    for variant in ("fever_local", "targeted_v2"):
        path = Path(f"evaluation_results/finreg_detector_manualeval_v3/{variant}/per_question.jsonl")
        for row in load_jsonl(path):
            row_id = row.get("id", "")
            if row_id and row_id not in questions:
                questions[row_id] = row.get("question", "")
    return questions


def main() -> None:
    args = parse_args()
    base_rows = load_csv(Path(args.base_review_csv))
    questions = build_question_map(Path(args.benchmark_prefill))
    existing = {row.get("id", ""): row for row in base_rows}

    merged_rows = list(base_rows)
    for row_id, data in SECOND_EXPANSION.items():
        if row_id in existing:
            continue
        merged_rows.append(
            {
                "id": row_id,
                "review_priority": data["review_priority"],
                "question": questions.get(row_id, ""),
                "suggested_label": data["suggested_label"],
                "suggested_error_type": data["suggested_error_type"],
                "likely_label": data["likely_label"],
                "likely_error_type": data["likely_error_type"],
                "reviewer_confidence": data["reviewer_confidence"],
                "review_note_draft": data["review_note_draft"],
                "gold_label": data["gold_label"],
                "gold_error_type": data["gold_error_type"],
                "review_notes": data["review_notes"],
            }
        )

    priority_order = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
    merged_rows.sort(key=lambda row: (priority_order.get(row.get("review_priority", "p9"), 9), row.get("id", "")))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Wrote {len(merged_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
