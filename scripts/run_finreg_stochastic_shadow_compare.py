#!/usr/bin/env python3
"""Run finreg-only shadow comparison across stochastic epistemic adapters.

This keeps the actual gate untouched and compares shadow epi sources under a
fixed 2D shadow policy, producing both per-run JSON outputs and one aggregate
report for ranking/inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finreg stochastic shadow comparison")
    parser.add_argument("--config", default="gating_finreg_ebcar_logit_mi_sc009_shadowfast")
    parser.add_argument(
        "--questions",
        default="data/domain_finreg/questions_finreg_conflict_50.jsonl",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epi-threshold", type=float, default=0.02)
    parser.add_argument("--ale-threshold", type=float, default=0.30)
    parser.add_argument(
        "--shadow-policy",
        default="epi_coupled_v2",
        choices=["legacy_ale_dominant", "epi_coupled_v1", "epi_coupled_v2"],
    )
    parser.add_argument(
        "--shadow-uncertainty-formula",
        default="v2_conflict_aware",
        choices=["v2_conflict_aware", "v3_decoupled"],
    )
    parser.add_argument(
        "--epi-sources",
        nargs="+",
        default=[
            "logit_mi",
            "stochastic_ou",
            "stochastic_langevin",
            "stochastic_mirror_langevin",
            "stochastic_wright_fisher",
            "stochastic_sghmc",
            "stochastic_sgbd",
            "stochastic_prox_langevin",
        ],
    )
    parser.add_argument(
        "--tmp-dir",
        default="evaluation_results/auto_eval/finreg_stochastic_compare_tmp",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results/auto_eval/finreg_stochastic_compare_20_seed7.json",
    )
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("HF_HOME", "./models/llm")
    env.setdefault("TRANSFORMERS_CACHE", "./models/llm")

    runs: list[dict[str, Any]] = []

    for epi_source in args.epi_sources:
        out_path = tmp_dir / (
            f"finreg_shadow_{epi_source}_limit{args.limit}_seed{args.seed}.json"
        )
        cmd = [
            "venv312/bin/python",
            "scripts/eval_grounding_proxy.py",
            "--config",
            args.config,
            "--questions",
            args.questions,
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--shadow-two-channel",
            "--shadow-epi-source",
            epi_source,
            "--shadow-policy",
            args.shadow_policy,
            "--shadow-uncertainty-formula",
            args.shadow_uncertainty_formula,
            "--uncertainty-threshold",
            str(args.epi_threshold),
            "--aleatoric-threshold",
            str(args.ale_threshold),
            "--output",
            str(out_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        payload = load_json(out_path)
        shadow = payload.get("shadow_two_channel", {}) or {}
        answer_stats = shadow.get("stats_answer", {}) or {}
        retrieve_stats = shadow.get("stats_retrieve_more", {}) or {}
        abstain_stats = shadow.get("stats_abstain", {}) or {}
        non_abstain_stats = shadow.get("stats_non_abstain", {}) or {}

        runs.append(
            {
                "epi_source": epi_source,
                "total": payload.get("total", 0),
                "actual_abstain_rate": payload.get("abstain_rate", 0.0),
                "shadow_answer_rate": shadow.get("answer_rate", 0.0),
                "shadow_retrieve_more_rate": shadow.get("retrieve_more_rate", 0.0),
                "shadow_abstain_rate": shadow.get("abstain_rate", 0.0),
                "shadow_actions": shadow.get("actions", {}),
                "answer_contradiction_rate": answer_stats.get("contradiction_rate"),
                "retrieve_contradiction_rate": retrieve_stats.get("contradiction_rate"),
                "abstain_contradiction_rate": abstain_stats.get("contradiction_rate"),
                "non_abstain_contradiction_rate": non_abstain_stats.get("contradiction_rate"),
                "answer_uncertainty_mean": answer_stats.get("uncertainty_mean"),
                "file": str(out_path),
            }
        )

    def rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
        answer_contra = item.get("answer_contradiction_rate")
        if answer_contra is None:
            answer_contra = 1.0
        return (
            float(answer_contra),
            float(item.get("shadow_abstain_rate", 1.0)),
            -float(item.get("shadow_answer_rate", 0.0)),
            float(item.get("non_abstain_contradiction_rate") or 1.0),
        )

    runs.sort(key=rank_key)

    report = {
        "config": args.config,
        "questions": args.questions,
        "limit": args.limit,
        "seed": args.seed,
        "epi_threshold": args.epi_threshold,
        "ale_threshold": args.ale_threshold,
        "shadow_policy": args.shadow_policy,
        "shadow_uncertainty_formula": args.shadow_uncertainty_formula,
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")


if __name__ == "__main__":
    main()
