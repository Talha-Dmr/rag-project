#!/usr/bin/env python3
"""
Build a curated Phase 2 seed-v2 review CSV by extending the existing reviewed
priority set with a conservative subset from the supported and hard-risk
shortlists.
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


CURATED_ADDITIONS: Dict[str, Dict[str, str]] = {
    "fever_local::fq01": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Core BCBS 239 objective answer appears grounded and directly responsive.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "directly grounded on BCBS 239 objective: accurate and timely risk data aggregation",
    },
    "fever_local::fq02": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Supervisory review answer is aligned with the retrieved governance-assessment statement.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "grounded summary of supervisory review assessing risk governance credibility under stress and normal operations",
    },
    "targeted_v2::fq16": {
        "review_priority": "p2",
        "suggested_label": "supported",
        "suggested_error_type": "none",
        "likely_label": "supported",
        "likely_error_type": "none",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer tracks the retrieved validation/governance differences closely enough to serve as a supported control.",
        "gold_label": "supported",
        "gold_error_type": "none",
        "review_notes": "supported comparative summary: frameworks vary in independent validation depth and governance intensity",
    },
    "targeted_v2::fq20": {
        "review_priority": "p2",
        "suggested_label": "partial",
        "suggested_error_type": "incomplete_reasoning",
        "likely_label": "partial",
        "likely_error_type": "incomplete_reasoning",
        "reviewer_confidence": "medium",
        "review_note_draft": "Main claim is plausible, but the answer backs away from the requested harmonization details.",
        "gold_label": "partial",
        "gold_error_type": "incomplete_reasoning",
        "review_notes": "partially grounded: suggests broad consistency but does not substantiate harmonization across guidance sets",
    },
    "targeted_v2::fq24": {
        "review_priority": "p2",
        "suggested_label": "partial",
        "suggested_error_type": "incomplete_reasoning",
        "likely_label": "partial",
        "likely_error_type": "incomplete_reasoning",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer captures common governance themes but overstates cross-body consistency.",
        "gold_label": "partial",
        "gold_error_type": "incomplete_reasoning",
        "review_notes": "partially supported: common board-accountability themes are grounded, but consistency is overstated",
    },
    "fever_local::fq34": {
        "review_priority": "p2",
        "suggested_label": "partial",
        "suggested_error_type": "incomplete_reasoning",
        "likely_label": "partial",
        "likely_error_type": "incomplete_reasoning",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer falls back to generic uncertainty handling instead of a concrete disagreement rationale.",
        "gold_label": "partial",
        "gold_error_type": "incomplete_reasoning",
        "review_notes": "partially grounded: generic abstention logic is present, but the stated justification for disagreement is underdeveloped",
    },
    "fever_local::fq29": {
        "review_priority": "p2",
        "suggested_label": "partial",
        "suggested_error_type": "incomplete_reasoning",
        "likely_label": "partial",
        "likely_error_type": "incomplete_reasoning",
        "reviewer_confidence": "medium",
        "review_note_draft": "Some grounded caution language, but the answer speculates about frequency differences without support.",
        "gold_label": "partial",
        "gold_error_type": "incomplete_reasoning",
        "review_notes": "partially supported: abstention theme is grounded, speculative claim about stricter BCBS frequency is not",
    },
    "fever_local::fq45": {
        "review_priority": "p1",
        "suggested_label": "partial",
        "suggested_error_type": "incomplete_reasoning",
        "likely_label": "partial",
        "likely_error_type": "incomplete_reasoning",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer mixes a direct yes/no claim with a later walk-back and never cleanly resolves the comparison.",
        "gold_label": "partial",
        "gold_error_type": "incomplete_reasoning",
        "review_notes": "internally inconsistent comparison: begins with a difference claim, then reverts to insufficient-context language",
    },
    "fever_local::fq46": {
        "review_priority": "p1",
        "suggested_label": "partial",
        "suggested_error_type": "incomplete_reasoning",
        "likely_label": "partial",
        "likely_error_type": "incomplete_reasoning",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer asserts a difference, then concedes the comparison is not supported by the context.",
        "gold_label": "partial",
        "gold_error_type": "incomplete_reasoning",
        "review_notes": "partial: starts with a concrete BCBS vs Fed claim, but the rest of the answer undermines that claim",
    },
    "fever_local::fq21": {
        "review_priority": "p1",
        "suggested_label": "partial",
        "suggested_error_type": "misinterpretation",
        "likely_label": "partial",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "medium",
        "review_note_draft": "Comparative explainability themes are present, but the entity mapping and rationale are muddled.",
        "gold_label": "partial",
        "gold_error_type": "misinterpretation",
        "review_notes": "partially grounded: answer identifies explainability dimensions but mixes institutions and does not cleanly map them",
    },
    "targeted_v2::fq14": {
        "review_priority": "p1",
        "suggested_label": "unsupported",
        "suggested_error_type": "fabricated_fact",
        "likely_label": "unsupported",
        "likely_error_type": "fabricated_fact",
        "reviewer_confidence": "high",
        "review_note_draft": "Answer trails into unrelated meta text and never delivers a reliable comparative conclusion.",
        "gold_label": "unsupported",
        "gold_error_type": "fabricated_fact",
        "review_notes": "generation drift: appended meta text and incomplete comparison make the response unreliable",
    },
    "targeted_v2::fq41": {
        "review_priority": "p1",
        "suggested_label": "unsupported",
        "suggested_error_type": "fabricated_fact",
        "likely_label": "unsupported",
        "likely_error_type": "fabricated_fact",
        "reviewer_confidence": "high",
        "review_note_draft": "Shortlist candidate contains clear non-domain contamination and should not be treated as grounded.",
        "gold_label": "unsupported",
        "gold_error_type": "fabricated_fact",
        "review_notes": "generation contamination: answer includes unrelated assistant-style text and fails the requested comparison",
    },
    "targeted_v2::fq26": {
        "review_priority": "p1",
        "suggested_label": "ambiguous",
        "suggested_error_type": "cross_document_conflict",
        "likely_label": "ambiguous",
        "likely_error_type": "cross_document_conflict",
        "reviewer_confidence": "medium",
        "review_note_draft": "Abstaining answer is mostly appropriate; the context suggests a difference but not a crisp confidence-safe comparison.",
        "gold_label": "ambiguous",
        "gold_error_type": "cross_document_conflict",
        "review_notes": "insufficiently resolvable comparison: answer ends in explicit uncertainty after surfacing partial differences",
    },
    "targeted_v2::fq43": {
        "review_priority": "p1",
        "suggested_label": "ambiguous",
        "suggested_error_type": "cross_document_conflict",
        "likely_label": "ambiguous",
        "likely_error_type": "cross_document_conflict",
        "reviewer_confidence": "medium",
        "review_note_draft": "Clean abstention to a 'which is more conservative' question that the retrieved evidence does not settle.",
        "gold_label": "ambiguous",
        "gold_error_type": "cross_document_conflict",
        "review_notes": "reasonable abstention: context does not cleanly determine which source is more conservative",
    },
    "targeted_v2::fq28": {
        "review_priority": "p1",
        "suggested_label": "ambiguous",
        "suggested_error_type": "cross_document_conflict",
        "likely_label": "ambiguous",
        "likely_error_type": "cross_document_conflict",
        "reviewer_confidence": "medium",
        "review_note_draft": "Abstaining comparison appears appropriate because the context does not rank BCBS vs EBA conservatism cleanly.",
        "gold_label": "ambiguous",
        "gold_error_type": "cross_document_conflict",
        "review_notes": "which-is-more-conservative comparison is not directly resolved by the retrieved evidence",
    },
    "fever_local::fq33": {
        "review_priority": "p1",
        "suggested_label": "ambiguous",
        "suggested_error_type": "cross_document_conflict",
        "likely_label": "ambiguous",
        "likely_error_type": "cross_document_conflict",
        "reviewer_confidence": "medium",
        "review_note_draft": "Reasonable uncertainty handling for a comparison that lacks a direct ranking signal.",
        "gold_label": "ambiguous",
        "gold_error_type": "cross_document_conflict",
        "review_notes": "ambiguous comparison: retrieved material does not clearly establish which source is more conservative",
    },
    "fever_local::fq31": {
        "review_priority": "p1",
        "suggested_label": "ambiguous",
        "suggested_error_type": "cross_document_conflict",
        "likely_label": "ambiguous",
        "likely_error_type": "cross_document_conflict",
        "reviewer_confidence": "medium",
        "review_note_draft": "Uncertainty is acceptable here because evidence for BCBS vs EBA outsourcing-control differences is incomplete.",
        "gold_label": "ambiguous",
        "gold_error_type": "cross_document_conflict",
        "review_notes": "context hints at differing standards but does not fully resolve the comparison under low-certainty conditions",
    },
    "fever_local::fq38": {
        "review_priority": "p1",
        "suggested_label": "ambiguous",
        "suggested_error_type": "cross_document_conflict",
        "likely_label": "ambiguous",
        "likely_error_type": "cross_document_conflict",
        "reviewer_confidence": "medium",
        "review_note_draft": "Another 'which is more conservative' comparison that is not directly answerable from the retrieved evidence.",
        "gold_label": "ambiguous",
        "gold_error_type": "cross_document_conflict",
        "review_notes": "insufficient evidence to rank BCBS vs ECB conservatism on manual adjustments",
    },
    "fever_local::fq11": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer says there is no direct disagreement, but the retrieved sources surface materially different timeliness expectations.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "contradicted by retrieved conflict: EBA proportional capability, PRA incident-driven prioritization, and BCBS rapid stress-event views differ",
    },
    "targeted_v2::fq42": {
        "review_priority": "p1",
        "suggested_label": "contradicted",
        "suggested_error_type": "misinterpretation",
        "likely_label": "contradicted",
        "likely_error_type": "misinterpretation",
        "reviewer_confidence": "medium",
        "review_note_draft": "Answer claims alignment while drifting to the wrong regulator and unsupported threshold details.",
        "gold_label": "contradicted",
        "gold_error_type": "misinterpretation",
        "review_notes": "false comparative claim: response answers a BCBS/ECB question with BCBS/EBA threshold language and overstates alignment",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate FinReg Phase 2 seed-v2 review CSV")
    parser.add_argument(
        "--base-review-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_draft.csv",
        help="Existing reviewed CSV to extend",
    )
    parser.add_argument(
        "--benchmark-prefill",
        default="evaluation_results/finreg_detector_phase2_prefill_smoke/benchmark_prefill.jsonl",
        help="Benchmark prefill used as a fallback source of questions",
    )
    parser.add_argument(
        "--supported-shortlist",
        default="evaluation_results/finreg_detector_phase2_supported_shortlist.csv",
        help="Supported shortlist CSV",
    )
    parser.add_argument(
        "--hard-risk-shortlist",
        default="evaluation_results/finreg_detector_phase2_hard_risk_shortlist.csv",
        help="Hard-risk shortlist CSV",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_seed_v2.csv",
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


def build_question_map(args: argparse.Namespace) -> Dict[str, str]:
    questions: Dict[str, str] = {}

    for row in load_csv(Path(args.supported_shortlist)):
        row_id = row.get("id", "")
        if row_id:
            questions[row_id] = row.get("question", "")

    for row in load_csv(Path(args.hard_risk_shortlist)):
        row_id = row.get("id", "")
        if row_id:
            questions[row_id] = row.get("question", "")

    for row in load_jsonl(Path(args.benchmark_prefill)):
        provenance = row.get("provenance") or {}
        source_id = provenance.get("source_id", "")
        if source_id and source_id not in questions:
            questions[source_id] = row.get("question", "")

    return questions


def main() -> None:
    args = parse_args()
    base_rows = load_csv(Path(args.base_review_csv))
    questions = build_question_map(args)
    existing = {row.get("id", ""): row for row in base_rows}

    merged_rows = list(base_rows)
    for row_id, data in CURATED_ADDITIONS.items():
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
