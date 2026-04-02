#!/usr/bin/env python3
"""Convenience test runner for the epistemic/Langevin research workflow."""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config_loader import load_config

DOMAIN_CONFIGS = {
    "health": "gating_health_ebcar_logit_mi_sc009",
    "finreg": "gating_finreg_ebcar_logit_mi_sc009_shadowfast",
    "disaster": "gating_disaster_ebcar_logit_mi_sc009",
}

DOMAIN_QUESTION_FILES = {
    "health": "data/domain_health/questions_health_conflict.jsonl",
    "finreg": "data/domain_finreg/questions_finreg_conflict.jsonl",
    "disaster": "data/domain_disaster/questions_disaster_conflict.jsonl",
}

SMOKE_DETECTOR_FALLBACK = {
    "model_path": "src/electra_daberta/final_fever_deberta_v3_base_model",
    "base_model": "microsoft/deberta-v3-base",
}


def run(cmd: Sequence[str], *, label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def ensure_files(paths: Iterable[Path], *, label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print(f"\nMissing expected files after {label}:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)


def ensure_smoke_prereqs(domain: str) -> None:
    config_name = DOMAIN_CONFIGS[domain]
    config = load_config(config_name)
    detector_cfg = (config.get("hallucination_detector") or {})
    model_path = detector_cfg.get("model_path")
    if not model_path:
        print(f"\nConfig `{config_name}` does not define `hallucination_detector.model_path`.")
        raise SystemExit(1)

    detector_path = ROOT / Path(str(model_path))
    if not detector_path.exists():
        fallback_path = ROOT / SMOKE_DETECTOR_FALLBACK["model_path"]
        if fallback_path.exists():
            print("\nPrimary detector checkpoint is missing; smoke test will use local FEVER fallback:")
            print(f"- missing:  {detector_path}")
            print(f"- fallback: {fallback_path}")
            return

        print("\nSmoke test prerequisites failed.")
        print(f"- Detector model path is missing: {detector_path}")
        print(f"- Fallback detector path is also missing: {fallback_path}")
        print("- The smoke test would run without hallucination detection and produce invalid epistemic outputs.")
        print("- Fix the config path or place a detector checkpoint there before rerunning.")
        raise SystemExit(1)


def run_unit() -> None:
    run(
        [sys.executable, "-m", "pytest", "tests/test_epistemic_research.py"],
        label="Unit Tests",
    )
    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "scripts/eval_grounding_proxy.py",
            "scripts/run_epistemic_shadow_compare.py",
            "scripts/render_epistemic_reports.py",
            "scripts/test_epistemic_program.py",
        ],
        label="Python Compile Check",
    )


def run_smoke(domain: str, limit: int, seed: int) -> None:
    ensure_smoke_prereqs(domain)
    config = load_config(DOMAIN_CONFIGS[domain])
    detector_cfg = (config.get("hallucination_detector") or {})
    configured_path = ROOT / Path(str(detector_cfg.get("model_path") or ""))
    use_fallback = not configured_path.exists()
    output_dir = ROOT / "evaluation_results" / "auto_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"smoke_{domain}_langevin.json"
    details_path = output_dir / f"smoke_{domain}_langevin.details.jsonl"

    cmd = [
        sys.executable,
        "scripts/eval_grounding_proxy.py",
        "--config",
        DOMAIN_CONFIGS[domain],
        "--questions",
        DOMAIN_QUESTION_FILES[domain],
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--shadow-two-channel",
        "--shadow-epi-source",
        "stochastic_langevin",
        "--shadow-policy",
        "epi_coupled_v2",
        "--shadow-uncertainty-formula",
        "v3_decoupled",
        "--uncertainty-threshold",
        "0.02",
        "--aleatoric-threshold",
        "0.30",
        "--output",
        str(summary_path),
        "--dump-details",
        str(details_path),
    ]
    if use_fallback:
        cmd.extend(
            [
                "--detector-model-path",
                SMOKE_DETECTOR_FALLBACK["model_path"],
                "--detector-base-model",
                SMOKE_DETECTOR_FALLBACK["base_model"],
            ]
        )

    run(cmd, label=f"Smoke Eval ({domain})")
    ensure_files([summary_path, details_path], label=f"smoke eval ({domain})")


def run_fast_matrix() -> None:
    summary_path = ROOT / "evaluation_results" / "auto_eval" / "epistemic_shadow_compare_summary.json"
    decision_matrix = ROOT / "docs" / "epistemic_decision_matrix.md"
    compare_report = ROOT / "docs" / "epistemic_shadow_compare.md"
    retrieve_more_report = ROOT / "docs" / "retrieve_more_utility_report.md"
    two_channel_note = ROOT / "docs" / "two_channel_shadow_note.md"

    run(
        [
            sys.executable,
            "scripts/run_epistemic_shadow_compare.py",
            "--skip-later-stages",
        ],
        label="Fast Matrix Compare",
    )
    run(
        [sys.executable, "scripts/render_epistemic_reports.py"],
        label="Render Canonical Reports",
    )
    ensure_files(
        [
            summary_path,
            decision_matrix,
            compare_report,
            retrieve_more_report,
            two_channel_note,
        ],
        label="fast matrix pipeline",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="unit",
        choices=["unit", "smoke", "fast-matrix", "all"],
    )
    parser.add_argument(
        "--domain",
        default="finreg",
        choices=["health", "finreg", "disaster"],
        help="Used only for --mode smoke or --mode all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Used only for --mode smoke or --mode all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Used only for --mode smoke or --mode all.",
    )
    args = parser.parse_args()

    if args.mode in {"unit", "all"}:
        run_unit()
    if args.mode in {"smoke", "all"}:
        run_smoke(args.domain, args.limit, args.seed)
    if args.mode in {"fast-matrix", "all"}:
        run_fast_matrix()

    print("\nEpistemic program test flow completed.")


if __name__ == "__main__":
    main()
