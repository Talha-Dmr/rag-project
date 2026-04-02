#!/usr/bin/env python3
"""Render canonical markdown reports from epistemic shadow comparison JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def candidate_stage_line(candidate: Dict[str, Any]) -> str:
    status = candidate.get("stage_status", {}) or {}
    return (
        f"| `{candidate['epi_source']}` | {status.get('stage_0', '-')} | "
        f"{status.get('stage_1', '-')} | {status.get('stage_2', '-')} | "
        f"{status.get('stage_3', '-')} | {status.get('stage_4', '-')} | "
        f"{candidate.get('final_decision', 'freeze')} |"
    )


def build_decision_matrix(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Epistemic Decision Matrix",
        "",
        "Canonical stage gates for the shadow epistemic program.",
        "",
        "## Stages",
        "",
        "1. `Stage 0`: feasibility on `20Q / seed=7`",
        "2. `Stage 1`: fast pass (`answered_contradiction_rate <= baseline`, abstain within band, contradiction guard)",
        "3. `Stage 2`: cost gate (runtime within allowed ratio vs baseline)",
        "4. `Stage 3`: confirmation on `50Q / seed=7`",
        "5. `Stage 4`: robustness on `50Q / seed=11,19`",
        "",
        "## Candidate Status",
        "",
        "| Candidate | S0 | S1 | S2 | S3 | S4 | Final |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in payload.get("candidates", []):
        lines.append(candidate_stage_line(candidate))
    lines.append("")
    return "\n".join(lines) + "\n"


def build_shadow_compare(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Epistemic Shadow Compare",
        "",
        f"Generated from `{payload.get('generated_at', 'unknown')}`.",
        "",
        "| Candidate | Final | Notes |",
        "| --- | --- | --- |",
    ]
    for candidate in payload.get("candidates", []):
        failures = []
        for run in candidate.get("runs", []):
            gate = run.get("gate_result", {}) or {}
            failures.extend(gate.get("reasons", []))
        notes = ", ".join(sorted(set(failures))) if failures else "passed all available gates"
        lines.append(
            f"| `{candidate['epi_source']}` | {candidate.get('final_decision', 'freeze')} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_retrieve_more_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Retrieve-More Utility Report",
        "",
        "This report points to per-run detail dumps emitted by `run_epistemic_shadow_compare.py`.",
        "",
        "| Candidate | Domain | Stage | Detail Dump |",
        "| --- | --- | --- | --- |",
    ]
    for candidate in payload.get("candidates", []):
        for run in candidate.get("runs", []):
            details_path = run.get("candidate_details_path")
            if not details_path:
                continue
            lines.append(
                f"| `{candidate['epi_source']}` | {run.get('domain', '-') } | "
                f"{run.get('stage_group', '-') } | `{details_path}` |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_two_channel_note(payload: Dict[str, Any]) -> str:
    settings = payload.get("settings", {}) or {}
    lines = [
        "# Two-Channel Shadow Note",
        "",
        "This note records the shadow-only 2D decision setup used during epistemic comparison.",
        "",
        f"- Shadow policy: `{settings.get('shadow_policy', 'unknown')}`",
        f"- Aleatoric formula: `{settings.get('shadow_uncertainty_formula', 'unknown')}`",
        f"- Epistemic threshold: `{settings.get('epi_threshold', 'unknown')}`",
        f"- Aleatoric threshold: `{settings.get('ale_threshold', 'unknown')}`",
        "",
        "The actual production gate remains unchanged while these comparisons are running.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="evaluation_results/auto_eval/epistemic_shadow_compare_summary.json",
    )
    parser.add_argument("--decision-matrix-out", default="docs/epistemic_decision_matrix.md")
    parser.add_argument("--compare-out", default="docs/epistemic_shadow_compare.md")
    parser.add_argument("--retrieve-more-out", default="docs/retrieve_more_utility_report.md")
    parser.add_argument("--two-channel-out", default="docs/two_channel_shadow_note.md")
    args = parser.parse_args()

    payload = load_json(ROOT / Path(args.input))
    outputs = {
        ROOT / Path(args.decision_matrix_out): build_decision_matrix(payload),
        ROOT / Path(args.compare_out): build_shadow_compare(payload),
        ROOT / Path(args.retrieve_more_out): build_retrieve_more_report(payload),
        ROOT / Path(args.two_channel_out): build_two_channel_note(payload),
    }
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
