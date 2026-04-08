#!/usr/bin/env python3
"""
Sanity check for the locked FinReg baseline.

This is intentionally lightweight: it does not run any models. It only checks that
the canonical config still matches the intended locked defaults.
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
    path = root / "config/gating_finreg_ebcar_logit_mi_sc009.yaml"

    if not path.exists():
        raise SystemExit(f"Missing config: {path}")

    cfg = load_yaml(path)

    expected = {
        "chunking.strategy": "section_aware",
        "retrieval.k": 20,
        "reranker.type": "ebcar",
        "gating.enabled": True,
        "gating.strategy": "retrieve_more",
        "gating.uncertainty_source": "logit_mi",
        "hallucination_detector.base_model": "microsoft/deberta-v3-base",
        "gating.contradiction_rate_threshold": 1.01,
    }

    failures: List[str] = []

    for k, v in expected.items():
        ok, msg = check_eq(cfg, k, v)
        if not ok:
            failures.append(msg)

    # Detector path must point to the FEVER DeBERTa-v3-base export.
    model_path = str(get(cfg, "hallucination_detector.model_path", default="") or "")
    if "final_fever_deberta_v3_base_model" not in model_path:
        failures.append(
            f"hallucination_detector.model_path should reference the FEVER DeBERTa-v3-base export, actual={model_path!r}"
        )

    if failures:
        print("LOCKED BASELINE CHECK: FAIL")
        for line in failures:
            print("-", line)
        raise SystemExit(1)

    print("LOCKED BASELINE CHECK: OK")
    print(f"- finreg: {path.relative_to(root)}")


if __name__ == "__main__":
    main()
