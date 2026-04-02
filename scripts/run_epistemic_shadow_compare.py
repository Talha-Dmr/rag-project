#!/usr/bin/env python3
"""Run staged shadow epistemic comparison across the high-stakes trio."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.epistemic_research import (
    DEFAULT_ABSTAIN_BOUNDS,
    DEFAULT_CONTRADICTION_GUARD,
    DEFAULT_RUNTIME_RATIO_LIMIT,
    evaluate_shadow_candidate,
    extract_shadow_run_metrics,
)


DEFAULT_DOMAIN_CONFIGS = {
    "health": "gating_health_ebcar_logit_mi_sc009",
    "finreg": "gating_finreg_ebcar_logit_mi_sc009_shadowfast",
    "disaster": "gating_disaster_ebcar_logit_mi_sc009",
}

DEFAULT_QUESTION_FILES = {
    "health": {
        20: "data/domain_health/questions_health_conflict.jsonl",
        50: "data/domain_health/questions_health_conflict_50.jsonl",
    },
    "finreg": {
        20: "data/domain_finreg/questions_finreg_conflict.jsonl",
        50: "data/domain_finreg/questions_finreg_conflict_50.jsonl",
    },
    "disaster": {
        20: "data/domain_disaster/questions_disaster_conflict.jsonl",
        50: "data/domain_disaster/questions_disaster_conflict_50.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["health", "finreg", "disaster"],
        choices=["health", "finreg", "disaster"],
    )
    parser.add_argument(
        "--epi-sources",
        nargs="+",
        default=[
            "logit_mi",
            "rep_mi",
            "mc_dropout",
            "swag",
            "stochastic_ou",
            "stochastic_langevin",
            "stochastic_mirror_langevin",
            "stochastic_wright_fisher",
        ],
    )
    parser.add_argument("--fast-limit", type=int, default=20)
    parser.add_argument("--confirm-limit", type=int, default=50)
    parser.add_argument("--fast-seed", type=int, default=7)
    parser.add_argument("--confirm-seed", type=int, default=7)
    parser.add_argument("--robust-seeds", default="11,19")
    parser.add_argument("--epi-threshold", type=float, default=0.02)
    parser.add_argument("--ale-threshold", type=float, default=0.30)
    parser.add_argument(
        "--shadow-policy",
        default="epi_coupled_v2",
        choices=["legacy_ale_dominant", "epi_coupled_v1", "epi_coupled_v2"],
    )
    parser.add_argument(
        "--shadow-uncertainty-formula",
        default="v3_decoupled",
        choices=["v2_conflict_aware", "v3_decoupled"],
    )
    parser.add_argument("--runtime-ratio-limit", type=float, default=DEFAULT_RUNTIME_RATIO_LIMIT)
    parser.add_argument(
        "--contradiction-guard",
        type=float,
        default=DEFAULT_CONTRADICTION_GUARD,
    )
    parser.add_argument(
        "--abstain-band",
        default=f"{DEFAULT_ABSTAIN_BOUNDS[0]},{DEFAULT_ABSTAIN_BOUNDS[1]}",
        help="Inclusive abstain-rate band for stage-1 pass, formatted as min,max.",
    )
    parser.add_argument(
        "--tmp-dir",
        default="evaluation_results/auto_eval/epistemic_shadow_compare",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results/auto_eval/epistemic_shadow_compare_summary.json",
    )
    parser.add_argument(
        "--skip-later-stages",
        action="store_true",
        help="Only run stage-0/1/2 fast pass (20Q seed=7).",
    )
    return parser.parse_args()


def parse_abstain_band(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--abstain-band must be formatted as min,max")
    return float(parts[0]), float(parts[1])


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_epi_source(epi_source: str) -> str:
    source = epi_source.strip().lower()
    mapping = {
        "rep_mi": "stochastic_langevin",
        "mc_dropout": "stochastic_ou",
        "swag": "stochastic_sghmc",
        "logit_mi": "logit_mi",
    }
    return mapping.get(source, source)


def build_eval_command(
    *,
    config_name: str,
    question_path: str,
    limit: int,
    seed: int,
    epi_source: str,
    summary_path: Path,
    details_path: Path,
    epi_threshold: float,
    ale_threshold: float,
    shadow_policy: str,
    shadow_uncertainty_formula: str,
) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "eval_grounding_proxy.py"),
        "--config",
        config_name,
        "--questions",
        question_path,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--shadow-two-channel",
        "--shadow-epi-source",
        normalize_epi_source(epi_source),
        "--shadow-policy",
        shadow_policy,
        "--shadow-uncertainty-formula",
        shadow_uncertainty_formula,
        "--uncertainty-threshold",
        str(epi_threshold),
        "--aleatoric-threshold",
        str(ale_threshold),
        "--output",
        str(summary_path),
        "--dump-details",
        str(details_path),
    ]


def run_eval(
    *,
    config_name: str,
    question_path: str,
    limit: int,
    seed: int,
    epi_source: str,
    tmp_dir: Path,
    epi_threshold: float,
    ale_threshold: float,
    shadow_policy: str,
    shadow_uncertainty_formula: str,
) -> Dict[str, Any]:
    safe_source = epi_source.replace("/", "_")
    stem = f"{config_name}_{safe_source}_{limit}_seed{seed}"
    summary_path = tmp_dir / f"{stem}.json"
    details_path = tmp_dir / f"{stem}.details.jsonl"
    cmd = build_eval_command(
        config_name=config_name,
        question_path=question_path,
        limit=limit,
        seed=seed,
        epi_source=epi_source,
        summary_path=summary_path,
        details_path=details_path,
        epi_threshold=epi_threshold,
        ale_threshold=ale_threshold,
        shadow_policy=shadow_policy,
        shadow_uncertainty_formula=shadow_uncertainty_formula,
    )
    completed = subprocess.run(cmd, capture_output=True, text=True)
    payload: Dict[str, Any] = {
        "ok": completed.returncode == 0,
        "summary_path": str(summary_path),
        "details_path": str(details_path),
        "command": cmd,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode == 0 and summary_path.exists():
        payload["summary"] = load_json(summary_path)
    return payload


def stage_label_for_seed(limit: int, seed: int, fast_limit: int, fast_seed: int, confirm_seed: int) -> str:
    if limit == fast_limit and seed == fast_seed:
        return "stage0_2_fast_pass"
    if limit != fast_limit and seed == confirm_seed:
        return "stage3_confirmation"
    return "stage4_robustness"


def main() -> None:
    args = parse_args()
    abstain_band = parse_abstain_band(args.abstain_band)
    robust_seeds = [int(seed.strip()) for seed in args.robust_seeds.split(",") if seed.strip()]
    tmp_dir = ROOT / Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, Any]] = []

    for epi_source in args.epi_sources:
        candidate_record: Dict[str, Any] = {
            "epi_source": epi_source,
            "runs": [],
            "stage_status": {
                "stage_0": "pending",
                "stage_1": "pending",
                "stage_2": "pending",
                "stage_3": "pending",
                "stage_4": "pending",
            },
            "final_decision": "freeze",
        }

        fast_pass = True
        fast_cost_pass = True
        for domain in args.domains:
            config_name = DEFAULT_DOMAIN_CONFIGS[domain]
            baseline_run = run_eval(
                config_name=config_name,
                question_path=DEFAULT_QUESTION_FILES[domain][args.fast_limit],
                limit=args.fast_limit,
                seed=args.fast_seed,
                epi_source="logit_mi",
                tmp_dir=tmp_dir,
                epi_threshold=args.epi_threshold,
                ale_threshold=args.ale_threshold,
                shadow_policy=args.shadow_policy,
                shadow_uncertainty_formula=args.shadow_uncertainty_formula,
            )
            candidate_run = run_eval(
                config_name=config_name,
                question_path=DEFAULT_QUESTION_FILES[domain][args.fast_limit],
                limit=args.fast_limit,
                seed=args.fast_seed,
                epi_source=epi_source,
                tmp_dir=tmp_dir,
                epi_threshold=args.epi_threshold,
                ale_threshold=args.ale_threshold,
                shadow_policy=args.shadow_policy,
                shadow_uncertainty_formula=args.shadow_uncertainty_formula,
            )

            if not baseline_run.get("ok") or not candidate_run.get("ok"):
                fast_pass = False
                fast_cost_pass = False
                candidate_record["stage_status"]["stage_0"] = "fail"
                candidate_record["runs"].append(
                    {
                        "domain": domain,
                        "limit": args.fast_limit,
                        "seed": args.fast_seed,
                        "candidate_pass_stage": "stage0_feasibility_failed",
                        "baseline": baseline_run,
                        "candidate": candidate_run,
                    }
                )
                continue

            candidate_record["stage_status"]["stage_0"] = "pass"
            baseline_metrics = extract_shadow_run_metrics(baseline_run["summary"])
            candidate_metrics = extract_shadow_run_metrics(candidate_run["summary"])
            gate = evaluate_shadow_candidate(
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_metrics,
                abstain_bounds=abstain_band,
                contradiction_guard=args.contradiction_guard,
                runtime_ratio_limit=args.runtime_ratio_limit,
            )
            candidate_record["runs"].append(
                {
                    "domain": domain,
                    "limit": args.fast_limit,
                    "seed": args.fast_seed,
                    "stage_group": stage_label_for_seed(
                        args.fast_limit,
                        args.fast_seed,
                        args.fast_limit,
                        args.fast_seed,
                        args.confirm_seed,
                    ),
                    "candidate_pass_stage": gate["candidate_pass_stage"],
                    "gate_result": gate,
                    "baseline_metrics": baseline_metrics,
                    "candidate_metrics": candidate_metrics,
                    "baseline_summary_path": baseline_run["summary_path"],
                    "candidate_summary_path": candidate_run["summary_path"],
                    "candidate_details_path": candidate_run["details_path"],
                }
            )
            fast_pass = fast_pass and gate["passed_stage_1"]
            fast_cost_pass = fast_cost_pass and gate["passed_stage_2"]

        candidate_record["stage_status"]["stage_1"] = "pass" if fast_pass else "fail"
        candidate_record["stage_status"]["stage_2"] = "pass" if fast_cost_pass else "fail"
        if args.skip_later_stages or not (fast_pass and fast_cost_pass):
            candidate_record["final_decision"] = "freeze"
            candidates.append(candidate_record)
            continue

        confirm_pass = True
        for domain in args.domains:
            config_name = DEFAULT_DOMAIN_CONFIGS[domain]
            baseline_run = run_eval(
                config_name=config_name,
                question_path=DEFAULT_QUESTION_FILES[domain][args.confirm_limit],
                limit=args.confirm_limit,
                seed=args.confirm_seed,
                epi_source="logit_mi",
                tmp_dir=tmp_dir,
                epi_threshold=args.epi_threshold,
                ale_threshold=args.ale_threshold,
                shadow_policy=args.shadow_policy,
                shadow_uncertainty_formula=args.shadow_uncertainty_formula,
            )
            candidate_run = run_eval(
                config_name=config_name,
                question_path=DEFAULT_QUESTION_FILES[domain][args.confirm_limit],
                limit=args.confirm_limit,
                seed=args.confirm_seed,
                epi_source=epi_source,
                tmp_dir=tmp_dir,
                epi_threshold=args.epi_threshold,
                ale_threshold=args.ale_threshold,
                shadow_policy=args.shadow_policy,
                shadow_uncertainty_formula=args.shadow_uncertainty_formula,
            )
            if not baseline_run.get("ok") or not candidate_run.get("ok"):
                confirm_pass = False
                candidate_record["runs"].append(
                    {
                        "domain": domain,
                        "limit": args.confirm_limit,
                        "seed": args.confirm_seed,
                        "stage_group": "stage3_confirmation",
                        "candidate_pass_stage": "stage3_confirmation_failed",
                        "baseline": baseline_run,
                        "candidate": candidate_run,
                    }
                )
                continue

            gate = evaluate_shadow_candidate(
                candidate_metrics=extract_shadow_run_metrics(candidate_run["summary"]),
                baseline_metrics=extract_shadow_run_metrics(baseline_run["summary"]),
                abstain_bounds=abstain_band,
                contradiction_guard=args.contradiction_guard,
                runtime_ratio_limit=args.runtime_ratio_limit,
            )
            gate["candidate_pass_stage"] = (
                "stage3_confirmation_pass"
                if gate["passed_stage_2"]
                else "stage3_confirmation_failed"
            )
            candidate_record["runs"].append(
                {
                    "domain": domain,
                    "limit": args.confirm_limit,
                    "seed": args.confirm_seed,
                    "stage_group": "stage3_confirmation",
                    "candidate_pass_stage": gate["candidate_pass_stage"],
                    "gate_result": gate,
                    "baseline_summary_path": baseline_run["summary_path"],
                    "candidate_summary_path": candidate_run["summary_path"],
                    "candidate_details_path": candidate_run["details_path"],
                }
            )
            confirm_pass = confirm_pass and gate["passed_stage_2"]

        candidate_record["stage_status"]["stage_3"] = "pass" if confirm_pass else "fail"
        if not confirm_pass:
            candidate_record["final_decision"] = "freeze"
            candidates.append(candidate_record)
            continue

        robustness_pass = True
        for seed in robust_seeds:
            for domain in args.domains:
                config_name = DEFAULT_DOMAIN_CONFIGS[domain]
                baseline_run = run_eval(
                    config_name=config_name,
                    question_path=DEFAULT_QUESTION_FILES[domain][args.confirm_limit],
                    limit=args.confirm_limit,
                    seed=seed,
                    epi_source="logit_mi",
                    tmp_dir=tmp_dir,
                    epi_threshold=args.epi_threshold,
                    ale_threshold=args.ale_threshold,
                    shadow_policy=args.shadow_policy,
                    shadow_uncertainty_formula=args.shadow_uncertainty_formula,
                )
                candidate_run = run_eval(
                    config_name=config_name,
                    question_path=DEFAULT_QUESTION_FILES[domain][args.confirm_limit],
                    limit=args.confirm_limit,
                    seed=seed,
                    epi_source=epi_source,
                    tmp_dir=tmp_dir,
                    epi_threshold=args.epi_threshold,
                    ale_threshold=args.ale_threshold,
                    shadow_policy=args.shadow_policy,
                    shadow_uncertainty_formula=args.shadow_uncertainty_formula,
                )
                if not baseline_run.get("ok") or not candidate_run.get("ok"):
                    robustness_pass = False
                    candidate_record["runs"].append(
                        {
                            "domain": domain,
                            "limit": args.confirm_limit,
                            "seed": seed,
                            "stage_group": "stage4_robustness",
                            "candidate_pass_stage": "stage4_robustness_failed",
                            "baseline": baseline_run,
                            "candidate": candidate_run,
                        }
                    )
                    continue

                gate = evaluate_shadow_candidate(
                    candidate_metrics=extract_shadow_run_metrics(candidate_run["summary"]),
                    baseline_metrics=extract_shadow_run_metrics(baseline_run["summary"]),
                    abstain_bounds=abstain_band,
                    contradiction_guard=args.contradiction_guard,
                    runtime_ratio_limit=args.runtime_ratio_limit,
                )
                gate["candidate_pass_stage"] = (
                    "stage4_robustness_pass"
                    if gate["passed_stage_2"]
                    else "stage4_robustness_failed"
                )
                candidate_record["runs"].append(
                    {
                        "domain": domain,
                        "limit": args.confirm_limit,
                        "seed": seed,
                        "stage_group": "stage4_robustness",
                        "candidate_pass_stage": gate["candidate_pass_stage"],
                        "gate_result": gate,
                        "baseline_summary_path": baseline_run["summary_path"],
                        "candidate_summary_path": candidate_run["summary_path"],
                        "candidate_details_path": candidate_run["details_path"],
                    }
                )
                robustness_pass = robustness_pass and gate["passed_stage_2"]

        candidate_record["stage_status"]["stage_4"] = "pass" if robustness_pass else "fail"
        candidate_record["final_decision"] = (
            "promote_candidate" if robustness_pass else "freeze"
        )
        candidates.append(candidate_record)

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "domains": args.domains,
        "epi_sources": args.epi_sources,
        "settings": {
            "fast_limit": args.fast_limit,
            "confirm_limit": args.confirm_limit,
            "fast_seed": args.fast_seed,
            "confirm_seed": args.confirm_seed,
            "robust_seeds": robust_seeds,
            "epi_threshold": args.epi_threshold,
            "ale_threshold": args.ale_threshold,
            "shadow_policy": args.shadow_policy,
            "shadow_uncertainty_formula": args.shadow_uncertainty_formula,
            "runtime_ratio_limit": args.runtime_ratio_limit,
            "contradiction_guard": args.contradiction_guard,
            "abstain_band": abstain_band,
        },
        "candidates": candidates,
    }

    output_path = ROOT / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote epistemic shadow comparison to {output_path}")


if __name__ == "__main__":
    main()
