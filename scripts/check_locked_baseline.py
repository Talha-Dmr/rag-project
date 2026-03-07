#!/usr/bin/env python3
"""
Sanity check for the locked high-stakes baseline.

This is intentionally lightweight: it does not run any models. It only checks that
the canonical high-stakes configs still match the intended locked defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} did not parse into a dict")
    return data


def get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default


def check_eq(cfg: Dict[str, Any], key: str, expected: Any) -> Tuple[bool, str]:
    actual = get(cfg, key, default=None)
    ok = actual == expected
    msg = f"{key}: expected={expected!r} actual={actual!r}"
    return ok, msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=".",
        help="Repo root (default: current directory).",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    configs = {
        "health": root / "config/gating_health_ebcar_logit_mi_sc009.yaml",
        "disaster": root / "config/gating_disaster_ebcar_logit_mi_sc009.yaml",
        "finreg": root / "config/gating_finreg_ebcar_logit_mi_sc009.yaml",
    }

    # Canonical locked expectations.
    expected = {
        "reranker.type": "ebcar",
        "gating.enabled": True,
        "gating.strategy": "retrieve_more",
        "gating.uncertainty_source": "logit_mi",
        "hallucination_detector.base_model": "microsoft/deberta-v3-small",
    }
    expected_domain = {
        "health": {"gating.contradiction_rate_threshold": 0.40},
        "finreg": {"gating.contradiction_rate_threshold": 0.40},
        "disaster": {"gating.contradiction_rate_threshold": 1.01},
    }

    failures: List[str] = []
    for domain, path in configs.items():
        if not path.exists():
            failures.append(f"[{domain}] missing: {path}")
            continue
        cfg = load_yaml(path)

        # Shared expectations.
        for k, v in expected.items():
            ok, msg = check_eq(cfg, k, v)
            if not ok:
                failures.append(f"[{domain}] {msg}")

        # Detector path must point to the balanced checkpoint family.
        model_path = str(get(cfg, "hallucination_detector.model_path", default="") or "")
        if "balanced" not in model_path:
            failures.append(
                f"[{domain}] hallucination_detector.model_path should reference balanced checkpoint, actual={model_path!r}"
            )

        # Domain-specific expectations.
        for k, v in expected_domain[domain].items():
            ok, msg = check_eq(cfg, k, v)
            if not ok:
                failures.append(f"[{domain}] {msg}")

    if failures:
        print("LOCKED BASELINE CHECK: FAIL")
        for line in failures:
            print("-", line)
        raise SystemExit(1)

    print("LOCKED BASELINE CHECK: OK")
    for domain, path in configs.items():
        print(f"- {domain}: {path.relative_to(root)}")


if __name__ == "__main__":
    main()

