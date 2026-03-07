#!/usr/bin/env python3
"""
Proxy grounding eval using hallucination-detector stats.

Outputs mean contradiction/uncertainty/source-consistency over all/answered/abstained.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def seed_everything(seed: int) -> None:
    # Keep eval runs reproducible even when the LLM uses sampling.
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism; some kernels may still be nondeterministic.
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            from transformers import set_seed  # type: ignore
            set_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def get_precheck_index_count(config: Dict) -> int | None:
    """
    Fast index-count check without loading full RAG stack.

    Returns:
      - int count for supported stores
      - None when precheck is unavailable
    """
    vector_cfg = config.get("vector_store", {}) or {}
    store_type = vector_cfg.get("type", "")
    vector_store_config = vector_cfg.get("config", {}) or {}
    if store_type != "chroma":
        return None

    persist_directory = vector_store_config.get("persist_directory") or vector_cfg.get("persist_directory")
    collection_name = vector_store_config.get("collection_name") or vector_cfg.get("collection_name") or "documents"
    if not persist_directory:
        return None

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
        return int(collection.count())
    except Exception:
        return 0


def load_questions(path: Path) -> List[Dict]:
    questions: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def update_stats(store: Dict[str, List[float]], stats: Dict[str, float]) -> None:
    for key in (
        "contradiction_rate",
        "contradiction_prob_mean",
        "uncertainty_mean",
        "source_consistency",
        "retrieval_max_score",
        "retrieval_mean_score",
        "entailment_prob_mean",
        "neutral_prob_mean",
        "contradiction_prob_std",
        "conflict_mass_mean",
        "label_entropy",
        "label_disagreement",
    ):
        val = stats.get(key)
        if isinstance(val, (int, float)):
            store[key].append(float(val))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(store: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: mean(v) for k, v in store.items()}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_shadow_uncertainty(stats: Dict[str, float]) -> Dict[str, float]:
    contradiction_prob = _clamp01(_to_float(stats.get("contradiction_prob_mean"), 0.0))
    source_consistency = _clamp01(_to_float(stats.get("source_consistency"), 0.0))
    retrieval_max = _clamp01(_to_float(stats.get("retrieval_max_score"), 0.0))
    retrieval_mean = _clamp01(_to_float(stats.get("retrieval_mean_score"), 0.0))
    retrieval_spread = _clamp01(max(0.0, retrieval_max - retrieval_mean))
    u_epi = max(0.0, _to_float(stats.get("uncertainty_mean"), 0.0))
    neutral_prob_mean = _clamp01(_to_float(stats.get("neutral_prob_mean"), 0.0))
    contradiction_prob_std = _clamp01(_to_float(stats.get("contradiction_prob_std"), 0.0))
    conflict_mass_mean = _clamp01(_to_float(stats.get("conflict_mass_mean"), 0.0))
    label_disagreement = _clamp01(_to_float(stats.get("label_disagreement"), 0.0))

    has_advanced = any(
        key in stats for key in (
            "neutral_prob_mean",
            "contradiction_prob_std",
            "conflict_mass_mean",
            "label_disagreement",
        )
    )
    if has_advanced:
        # Conflict-aware aleatoric proxy (v2):
        # - contradiction probability
        # - label disagreement across contexts
        # - neutral mass (ambiguity)
        # - entailment/contradiction overlap mass
        # - contradiction spread
        # - retrieval spread (minor)
        u_ale = _clamp01(
            0.35 * contradiction_prob
            + 0.20 * label_disagreement
            + 0.15 * neutral_prob_mean
            + 0.15 * conflict_mass_mean
            + 0.10 * contradiction_prob_std
            + 0.05 * retrieval_spread
        )
    else:
        # Backward-compatible fallback.
        u_ale = _clamp01((contradiction_prob + (1.0 - source_consistency) + retrieval_spread) / 3.0)

    return {
        "u_epi": u_epi,
        "u_ale": u_ale,
        "retrieval_spread": retrieval_spread,
        "contradiction_prob": contradiction_prob,
        "source_consistency": source_consistency,
        "neutral_prob_mean": neutral_prob_mean,
        "contradiction_prob_std": contradiction_prob_std,
        "conflict_mass_mean": conflict_mass_mean,
        "label_disagreement": label_disagreement,
    }


def compute_shadow_epistemic(stats: Dict[str, float], u_epi_baseline: float, epi_source: str) -> float:
    """
    Shadow epistemic adapter.

    This does not affect actual gate decisions. It enables controlled comparison of
    baseline epistemic vs stochastic-style epistemic proxies in shadow mode.
    """
    source = (epi_source or "logit_mi").strip().lower()
    base = max(0.0, float(u_epi_baseline))
    if source in ("logit_mi", "baseline"):
        return base

    contradiction_prob = _clamp01(_to_float(stats.get("contradiction_prob_mean"), 0.0))
    contradiction_prob_std = _clamp01(_to_float(stats.get("contradiction_prob_std"), 0.0))
    label_disagreement = _clamp01(_to_float(stats.get("label_disagreement"), 0.0))
    neutral_prob_mean = _clamp01(_to_float(stats.get("neutral_prob_mean"), 0.0))
    retrieval_spread = _clamp01(
        max(
            0.0,
            _to_float(stats.get("retrieval_max_score"), 0.0)
            - _to_float(stats.get("retrieval_mean_score"), 0.0),
        )
    )

    if source in ("stochastic_ou", "ou"):
        # OU-like smooth proxy: keep inertia on baseline epistemic, add mild noise-scale factors.
        return max(
            0.0,
            0.80 * base
            + 0.10 * contradiction_prob_std
            + 0.07 * label_disagreement
            + 0.03 * retrieval_spread,
        )

    if source in ("stochastic_langevin", "langevin"):
        # Langevin-like proxy: allow stronger pull from conflict-related gradients.
        return max(
            0.0,
            0.60 * base
            + 0.20 * contradiction_prob
            + 0.12 * label_disagreement
            + 0.08 * neutral_prob_mean,
        )

    if source in ("stochastic_mirror_langevin", "mirror_langevin", "mla"):
        # Mirror-Langevin-like proxy:
        # - Use simplex-aware entropy normalization as mirror geometry signal.
        # - Reward disagreement/spread while keeping baseline inertia.
        entailment_prob_mean = _clamp01(_to_float(stats.get("entailment_prob_mean"), 0.0))
        total_prob = max(1e-8, entailment_prob_mean + neutral_prob_mean + contradiction_prob)
        p_e = entailment_prob_mean / total_prob
        p_n = neutral_prob_mean / total_prob
        p_c = contradiction_prob / total_prob
        entropy = 0.0
        for p in (p_e, p_n, p_c):
            entropy -= p * math.log(max(1e-8, p))
        entropy_norm = _clamp01(entropy / math.log(3.0))
        mirror_tension = 1.0 - entropy_norm
        return max(
            0.0,
            0.58 * base
            + 0.16 * contradiction_prob_std
            + 0.14 * label_disagreement
            + 0.12 * mirror_tension,
        )

    if source in ("stochastic_wright_fisher", "wright_fisher", "wf"):
        # Wright-Fisher-like proxy:
        # - Diffusion strength is higher near simplex interior and lower near corners.
        # - Combine with disagreement to keep conflict-awareness.
        entailment_prob_mean = _clamp01(_to_float(stats.get("entailment_prob_mean"), 0.0))
        total_prob = max(1e-8, entailment_prob_mean + neutral_prob_mean + contradiction_prob)
        p_e = entailment_prob_mean / total_prob
        p_n = neutral_prob_mean / total_prob
        p_c = contradiction_prob / total_prob
        wf_strength = (p_e * (1.0 - p_e) + p_n * (1.0 - p_n) + p_c * (1.0 - p_c)) / 3.0
        wf_strength = _clamp01(wf_strength / 0.25)
        return max(
            0.0,
            0.52 * base
            + 0.26 * wf_strength
            + 0.14 * label_disagreement
            + 0.08 * contradiction_prob_std,
        )

    return base


def decide_shadow_action_2d(u_epi: float, u_ale: float, epi_threshold: float, ale_threshold: float) -> str:
    epi_high = u_epi >= epi_threshold
    ale_high = u_ale >= ale_threshold
    if epi_high and ale_high:
        return "abstain"
    if epi_high and not ale_high:
        return "retrieve_more"
    if (not epi_high) and ale_high:
        return "abstain"
    return "answer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy grounding eval")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--questions", required=True, help="Path to JSONL questions")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument(
        "--dump-details",
        default="",
        help="Optional JSONL path to dump per-question gating details (debug/repro aid).",
    )
    parser.add_argument(
        "--dump-include-answer",
        action="store_true",
        help="Include a truncated answer preview in --dump-details output.",
    )
    parser.add_argument("--contradiction-rate-threshold", type=float, default=None)
    parser.add_argument("--contradiction-prob-threshold", type=float, default=None)
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    parser.add_argument("--source-consistency-threshold", type=float, default=None)
    parser.add_argument("--uncertainty-source", default=None)
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Optional override for config.llm.temperature (use 0.0 for deterministic eval).",
    )
    parser.add_argument(
        "--allow-empty-index",
        action="store_true",
        help="Run even if vector collection is empty (not recommended).",
    )
    parser.add_argument(
        "--shadow-two-channel",
        action="store_true",
        help="Compute 2D shadow policy using u_epi/u_ale without affecting actual gate action.",
    )
    parser.add_argument(
        "--aleatoric-threshold",
        type=float,
        default=None,
        help="Optional threshold for shadow u_ale. If omitted, derived from config thresholds.",
    )
    parser.add_argument(
        "--shadow-epi-source",
        default="logit_mi",
        choices=[
            "logit_mi",
            "stochastic_ou",
            "stochastic_langevin",
            "stochastic_mirror_langevin",
            "stochastic_wright_fisher",
        ],
        help="Shadow epistemic source adapter (does not affect actual gate decisions).",
    )
    args = parser.parse_args()

    seed_everything(int(args.seed))

    questions = load_questions(Path(args.questions))
    if args.limit:
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.limit]

    config = load_config(args.config)
    if args.llm_temperature is not None:
        llm_cfg = config.get("llm") or {}
        if not isinstance(llm_cfg, dict):
            llm_cfg = {}
        llm_cfg["temperature"] = float(args.llm_temperature)
        config["llm"] = llm_cfg
    precheck_count = get_precheck_index_count(config)
    if precheck_count == 0 and not args.allow_empty_index:
        raise SystemExit(
            "Vector collection is empty (0 docs). Index a corpus first or pass "
            "--allow-empty-index to override."
        )
    rag = RAGPipeline.from_config(config)
    index_count = rag.vector_store.get_count()
    if index_count == 0 and not args.allow_empty_index:
        raise SystemExit(
            "Vector collection is empty (0 docs). Index a corpus first or pass "
            "--allow-empty-index to override."
        )

    abstain_message = (
        config.get("gating", {}).get("abstain_message", "").strip()
    )

    gating_override = {}
    if args.contradiction_rate_threshold is not None:
        gating_override["contradiction_rate_threshold"] = args.contradiction_rate_threshold
    if args.contradiction_prob_threshold is not None:
        gating_override["contradiction_prob_threshold"] = args.contradiction_prob_threshold
    if args.uncertainty_threshold is not None:
        gating_override["uncertainty_threshold"] = args.uncertainty_threshold
    if args.source_consistency_threshold is not None:
        gating_override["source_consistency_threshold"] = args.source_consistency_threshold
    if args.uncertainty_source is not None:
        gating_override["uncertainty_source"] = args.uncertainty_source

    gating_cfg = config.get("gating", {}) or {}
    epi_threshold = float(
        gating_override.get(
            "uncertainty_threshold",
            gating_cfg.get("uncertainty_threshold", 0.0),
        )
    )
    if args.aleatoric_threshold is not None:
        ale_threshold = float(args.aleatoric_threshold)
    else:
        cfg_ale = gating_cfg.get("aleatoric_threshold")
        if isinstance(cfg_ale, (int, float)):
            ale_threshold = float(cfg_ale)
        else:
            # Reasonable shadow default for v2 aleatoric proxy.
            ale_threshold = 0.30

    action_counts = Counter()
    shadow_action_counts = Counter()
    total = 0
    abstain = 0
    shadow_abstain = 0
    detector_failures = 0
    details_fh = None
    if args.dump_details:
        details_path = Path(args.dump_details)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_fh = details_path.open("w", encoding="utf-8")

    buckets = {
        "all": defaultdict(list),
        "answered": defaultdict(list),
        "abstain": defaultdict(list),
    }
    shadow_buckets = {
        "answered": defaultdict(list),
        "abstain": defaultdict(list),
    }
    u_epi_values: List[float] = []
    u_epi_stochastic_values: List[float] = []
    u_ale_values: List[float] = []
    retrieval_spread_values: List[float] = []

    for item in questions:
        query = item.get("query", "")
        if not query:
            continue

        result = rag.query(
            query_text=query,
            return_context=False,
            detect_hallucinations=True,
            gating=gating_override if gating_override else None,
        )
        answer = (result.get("answer") or "").strip()
        gating = result.get("gating") or {}
        stats = gating.get("stats") or {}
        action = gating.get("action", "none")
        attempts = gating.get("attempts", 0)
        k_used = gating.get("k_used")
        if result.get("hallucination_detected") is None or result.get("hallucination_error"):
            detector_failures += 1

        action_counts[action] += 1
        total += 1

        is_abstain = bool(abstain_message and answer == abstain_message)
        if is_abstain:
            abstain += 1

        update_stats(buckets["all"], stats)
        update_stats(buckets["abstain" if is_abstain else "answered"], stats)

        shadow_metrics = compute_shadow_uncertainty(stats)
        u_epi = shadow_metrics["u_epi"]
        u_epi_stochastic = compute_shadow_epistemic(
            stats=stats,
            u_epi_baseline=u_epi,
            epi_source=args.shadow_epi_source,
        )
        u_ale = shadow_metrics["u_ale"]
        retrieval_spread = shadow_metrics["retrieval_spread"]
        u_epi_values.append(u_epi)
        u_epi_stochastic_values.append(u_epi_stochastic)
        u_ale_values.append(u_ale)
        retrieval_spread_values.append(retrieval_spread)

        shadow_action = ""
        if args.shadow_two_channel:
            shadow_action = decide_shadow_action_2d(
                u_epi=u_epi_stochastic,
                u_ale=u_ale,
                epi_threshold=epi_threshold,
                ale_threshold=ale_threshold,
            )
            shadow_action_counts[shadow_action] += 1
            is_shadow_abstain = shadow_action == "abstain"
            if is_shadow_abstain:
                shadow_abstain += 1
            update_stats(shadow_buckets["abstain" if is_shadow_abstain else "answered"], stats)

        if details_fh is not None:
            row = {
                "query": query,
                "action": action,
                "attempts": attempts,
                "k_used": k_used,
                "is_abstain": bool(is_abstain),
                "stats": stats,
                "u_epi": u_epi,
                "u_epi_baseline": u_epi,
                "u_epi_stochastic": u_epi_stochastic,
                "u_ale": u_ale,
                "retrieval_spread": retrieval_spread,
            }
            if args.shadow_two_channel:
                row["action_shadow_2d"] = shadow_action
            if "hallucination_detected" in result:
                row["hallucination_detected"] = result.get("hallucination_detected")
            if result.get("hallucination_error"):
                row["hallucination_error"] = result.get("hallucination_error")
            if args.dump_include_answer:
                preview = answer.replace("\n", " ").strip()
                row["answer_preview"] = preview[:400]
            details_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total": total,
        "abstain": abstain,
        "abstain_rate": (abstain / total) if total else 0.0,
        "detector_failures": detector_failures,
        "actions": dict(action_counts),
        "stats_all": summarize(buckets["all"]),
        "stats_answered": summarize(buckets["answered"]),
        "stats_abstain": summarize(buckets["abstain"]),
        "u_epi_mean": mean(u_epi_values),
        "u_epi_stochastic_mean": mean(u_epi_stochastic_values),
        "u_ale_mean": mean(u_ale_values),
        "retrieval_spread_mean": mean(retrieval_spread_values),
        "shadow_two_channel": {
            "enabled": bool(args.shadow_two_channel),
            "aleatoric_formula": "v2_conflict_aware",
            "epi_source": args.shadow_epi_source,
            "epi_threshold": epi_threshold,
            "ale_threshold": ale_threshold,
            "actions": dict(shadow_action_counts),
            "abstain": shadow_abstain,
            "abstain_rate": (shadow_abstain / total) if total else 0.0,
            "stats_answered": summarize(shadow_buckets["answered"]),
            "stats_abstain": summarize(shadow_buckets["abstain"]),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote summary to {out_path}")
    if details_fh is not None:
        details_fh.close()
        print(f"Wrote per-question details to {args.dump_details}")


if __name__ == "__main__":
    main()
