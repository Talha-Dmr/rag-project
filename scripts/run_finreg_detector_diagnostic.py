#!/usr/bin/env python3
"""
Run an isolated finreg detector diagnostic across multiple detector variants.

Design goals:
- keep retrieval / reranker / generation / gating fixed unless explicitly requested
- swap only the detector block by default
- emit per-question JSONL, per-detector summary JSON, comparison markdown
- build a stratified manual-eval subset and annotation sheet
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


DEFAULT_BASE_CONFIG = "gating_finreg_ebcar_logit_mi_sc009"
DEFAULT_QUESTION_SET = "data/domain_finreg/questions_finreg_conflict_50.jsonl"

DEFAULT_VARIANTS = {
    "fever_local": "gating_finreg_ebcar_logit_mi_sc009_localdet",
    "balanced": "gating_finreg_ebcar_logit_mi_sc009",
    "targeted_contraguard": "gating_finreg_ebcar_logit_mi_sc009_targetedcontraguarddet",
}

BUCKET_ORDER = (
    "high_entailment",
    "uncertain_zone",
    "high_contradiction_signal",
    "suspicious_cases",
)

ERROR_TYPES = (
    "fabricated_fact",
    "wrong_number_or_threshold",
    "cross_document_conflict",
    "outdated_regulation",
    "misinterpretation",
    "incomplete_reasoning",
)


def seed_everything(seed: int) -> None:
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


def load_questions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_variants(user_variants: List[str]) -> List[Tuple[str, str]]:
    if not user_variants:
        return list(DEFAULT_VARIANTS.items())

    resolved: List[Tuple[str, str]] = []
    for raw in user_variants:
        if "=" in raw:
            alias, config_name = raw.split("=", 1)
            resolved.append((alias.strip(), config_name.strip()))
            continue
        key = raw.strip()
        config_name = DEFAULT_VARIANTS.get(key, key)
        resolved.append((key, config_name))
    return resolved


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def make_histogram(values: List[float], bins: int = 10) -> Dict[str, Any]:
    counts = [0 for _ in range(bins)]
    for value in values:
        clamped = min(1.0, max(0.0, float(value)))
        idx = min(bins - 1, int(math.floor(clamped * bins)))
        if clamped == 1.0:
            idx = bins - 1
        counts[idx] += 1
    edges = []
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        edges.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": counts[idx]})
    return {
        "bins": bins,
        "counts": counts,
        "edges": edges,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean": _mean(values),
        "std": _std(values),
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "narrow_band_0_0_1": bool(values) and max(values) <= 0.1,
    }


def dominant_detector_label(record: Dict[str, Any]) -> str:
    scores = [
        ("entailment", _to_float(record.get("entailment_prob_mean"))),
        ("neutral", _to_float(record.get("neutral_prob_mean"))),
        ("contradiction", _to_float(record.get("contradiction_prob_mean"))),
    ]
    return max(scores, key=lambda item: item[1])[0]


def shorten(text: str, limit: int = 700) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit]


def extract_retrieved_chunks(context: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, doc in enumerate(context[:top_k], start=1):
        metadata = doc.get("metadata") or {}
        rows.append({
            "rank": idx,
            "score": _to_float(doc.get("score")),
            "content": doc.get("content", ""),
            "content_preview": shorten(doc.get("content", "")),
            "title": metadata.get("title"),
            "source": metadata.get("source"),
            "page": metadata.get("page") or metadata.get("page_number"),
            "section": metadata.get("section") or metadata.get("heading"),
            "chunk_id": metadata.get("chunk_id"),
        })
    return rows


def validate_isolation(base_config: Dict[str, Any], variant_config: Dict[str, Any]) -> Dict[str, bool]:
    def _section_equal(name: str) -> bool:
        return copy.deepcopy(base_config.get(name)) == copy.deepcopy(variant_config.get(name))

    return {
        "same_chunking": _section_equal("chunking"),
        "same_embeddings": _section_equal("embeddings"),
        "same_vector_store": _section_equal("vector_store"),
        "same_llm": _section_equal("llm"),
        "same_retrieval": _section_equal("retrieval"),
        "same_reranker": _section_equal("reranker"),
        "same_gating": _section_equal("gating"),
    }


def build_variant_config(
    base_config_name: str,
    detector_config_name: str,
    keep_source_gating: bool,
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    base_config = load_config(base_config_name)
    detector_config = load_config(detector_config_name)
    merged = copy.deepcopy(base_config)
    merged["hallucination_detector"] = copy.deepcopy(
        detector_config.get("hallucination_detector", {})
    )
    if keep_source_gating:
        merged["gating"] = copy.deepcopy(detector_config.get("gating", merged.get("gating", {})))
    isolation = validate_isolation(base_config, merged)
    return merged, isolation


def resolve_detector_model_path(config: Dict[str, Any]) -> Path | None:
    detector_cfg = config.get("hallucination_detector") or {}
    model_path = detector_cfg.get("model_path")
    if not model_path:
        return None
    candidate = Path(str(model_path))
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_detector_short_scores(row: Dict[str, Any]) -> str:
    return (
        f"E={_to_float(row.get('entailment_prob_mean')):.3f} | "
        f"N={_to_float(row.get('neutral_prob_mean')):.3f} | "
        f"C={_to_float(row.get('contradiction_prob_mean')):.3f} | "
        f"HTopK={_to_float(row.get('hallucination_prob_topk')):.3f} | "
        f"DConflict={_to_float(row.get('detector_conflict')):.3f}"
    )


def build_bucket(
    record: Dict[str, Any],
    *,
    suspicious_conflict_threshold: float,
    suspicious_hallucination_topk_threshold: float,
    contradiction_signal_threshold: float,
    high_contradiction_gap_threshold: float,
    uncertainty_gap_threshold: float,
    high_entailment_threshold: float,
    low_contradiction_threshold: float,
) -> str:
    entailment = _to_float(record.get("entailment_prob_mean"))
    neutral = _to_float(record.get("neutral_prob_mean"))
    contradiction = _to_float(record.get("contradiction_prob_mean"))
    hallucination_topk = _to_float(record.get("hallucination_prob_topk"))
    detector_conflict = _to_float(record.get("detector_conflict"))
    entailment_contradiction_gap = entailment - contradiction

    if (
        contradiction >= contradiction_signal_threshold
        and entailment_contradiction_gap <= high_contradiction_gap_threshold
    ):
        return "high_contradiction_signal"
    if (
        detector_conflict >= suspicious_conflict_threshold
        or hallucination_topk >= suspicious_hallucination_topk_threshold
    ):
        return "suspicious_cases"
    if abs(entailment - neutral) <= uncertainty_gap_threshold and contradiction <= max(entailment, neutral):
        return "uncertain_zone"
    if entailment >= high_entailment_threshold and contradiction <= low_contradiction_threshold:
        return "high_entailment"
    if contradiction > entailment:
        return "high_contradiction_signal"
    return "uncertain_zone"


def sample_stratified_subset(
    records: List[Dict[str, Any]],
    total_target: int,
    seed: int,
    *,
    suspicious_conflict_threshold: float,
    suspicious_hallucination_topk_threshold: float,
    contradiction_signal_threshold: float,
    high_contradiction_gap_threshold: float,
    uncertainty_gap_threshold: float,
    high_entailment_threshold: float,
    low_contradiction_threshold: float,
) -> List[Dict[str, Any]]:
    by_bucket: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    for row in records:
        bucket = build_bucket(
            row,
            suspicious_conflict_threshold=suspicious_conflict_threshold,
            suspicious_hallucination_topk_threshold=suspicious_hallucination_topk_threshold,
            contradiction_signal_threshold=contradiction_signal_threshold,
            high_contradiction_gap_threshold=high_contradiction_gap_threshold,
            uncertainty_gap_threshold=uncertainty_gap_threshold,
            high_entailment_threshold=high_entailment_threshold,
            low_contradiction_threshold=low_contradiction_threshold,
        )
        row["bucket"] = bucket
        by_bucket[bucket].append(row)

    rng = random.Random(seed)
    for rows in by_bucket.values():
        rng.shuffle(rows)

    per_bucket_target = max(1, total_target // len(BUCKET_ORDER))
    selected: List[Dict[str, Any]] = []
    seen_ids = set()

    for bucket in BUCKET_ORDER:
        for row in by_bucket[bucket]:
            unique_id = row["id"]
            if unique_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(unique_id)
            if sum(1 for item in selected if item["bucket"] == bucket) >= per_bucket_target:
                break

    leftovers = []
    for bucket in BUCKET_ORDER:
        leftovers.extend(row for row in by_bucket[bucket] if row["id"] not in seen_ids)
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= total_target:
            break
        selected.append(row)
        seen_ids.add(row["id"])

    return selected


def build_annotation_rows(subset: List[Dict[str, Any]], top_k_contexts: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in subset:
        contexts = row.get("retrieved_chunks", [])[:top_k_contexts]
        context_text = "\n\n".join(
            f"[{chunk['rank']}] {chunk.get('source') or chunk.get('title') or 'context'} | "
            f"score={_to_float(chunk.get('score')):.3f}\n{chunk.get('content_preview', '')}"
            for chunk in contexts
        )
        rows.append({
            "id": row["id"],
            "question_id": row["question_id"],
            "detector_variant": row["detector_variant"],
            "bucket": row["bucket"],
            "question": row["question"],
            "generated_answer": row["generated_answer"],
            "retrieved_context": context_text,
            "retrieval_max_score": f"{_to_float(row.get('retrieval_max_score')):.3f}",
            "retrieval_mean_score": f"{_to_float(row.get('retrieval_mean_score')):.3f}",
            "detector_scores": build_detector_short_scores(row),
            "predicted_detector_label": row["predicted_detector_label"],
            "label": "",
            "error_type": "",
            "notes": "",
        })
    return rows


def write_annotation_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question_id",
        "detector_variant",
        "bucket",
        "question",
        "generated_answer",
        "retrieved_context",
        "retrieval_max_score",
        "retrieval_mean_score",
        "detector_scores",
        "predicted_detector_label",
        "label",
        "error_type",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_report(
    output_dir: Path,
    base_config_name: str,
    question_path: Path,
    variants: List[Tuple[str, str]],
    summaries: Dict[str, Dict[str, Any]],
    subset_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Finreg Detector Comparison Report")
    lines.append("")
    lines.append("## Run Setup")
    lines.append("")
    lines.append(f"- Base config: `{base_config_name}`")
    lines.append(f"- Questions: `{question_path}`")
    lines.append(f"- Variants: {', '.join(f'`{alias}` -> `{cfg}`' for alias, cfg in variants)}")
    lines.append("- Non-detector settings are inherited from the base config unless `--use-source-gating` is set.")
    lines.append("")
    lines.append("## Detector Summary")
    lines.append("")
    lines.append("| Variant | Answered Rate | Abstain Rate | Contradiction Rate | Unsupported Answer Rate | Global C Mean | Global E Mean | Dominant Contradiction Count | Conflict Trigger Count |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for alias, _ in variants:
        summary = summaries[alias]
        status = "invalid" if not summary.get("detector_available", True) else "ok"
        lines.append(
            "| "
            + f"{alias} ({status}) | "
            + f"{summary['answered_rate']:.3f} | "
            + f"{summary['abstain_rate']:.3f} | "
            + f"{summary['contradiction_rate']:.3f} | "
            + f"{summary['unsupported_answer_rate']:.3f} | "
            + f"{summary['contradiction_prob_mean_global']:.3f} | "
            + f"{summary['entailment_prob_mean_global']:.3f} | "
            + f"{summary['distribution_analysis']['dominant_contradiction_count']} | "
            + f"{summary['distribution_analysis']['detector_conflict_trigger_count']} |"
        )
    lines.append("")
    lines.append("## Score Distribution Diagnostics")
    lines.append("")
    for alias, _ in variants:
        summary = summaries[alias]
        dist = summary["distribution_analysis"]
        lines.append(f"### {alias}")
        lines.append("")
        lines.append(f"- Detector available: `{summary.get('detector_available', True)}`")
        lines.append(f"- Detector model path: `{summary.get('detector_model_path')}`")
        if summary.get("detector_error"):
            lines.append(f"- Detector error: `{summary['detector_error']}`")
        lines.append(
            f"- Contradiction spread: min={dist['contradiction_prob_histogram']['min']:.3f}, "
            f"max={dist['contradiction_prob_histogram']['max']:.3f}, "
            f"mean={dist['contradiction_prob_histogram']['mean']:.3f}, "
            f"std={dist['contradiction_prob_histogram']['std']:.3f}"
        )
        lines.append(
            f"- Entailment spread: min={dist['entailment_prob_histogram']['min']:.3f}, "
            f"max={dist['entailment_prob_histogram']['max']:.3f}, "
            f"mean={dist['entailment_prob_histogram']['mean']:.3f}, "
            f"std={dist['entailment_prob_histogram']['std']:.3f}"
        )
        lines.append(
            f"- Narrow contradiction band (0.0-0.1 only): "
            f"`{dist['contradiction_prob_histogram']['narrow_band_0_0_1']}`"
        )
        lines.append(
            f"- Dominant contradiction class count: `{dist['dominant_contradiction_count']}` / "
            f"`{summary['total_questions']}`"
        )
        lines.append(
            f"- detector_conflict > 0 count: `{dist['detector_conflict_trigger_count']}` / "
            f"`{summary['total_questions']}`"
        )
        lines.append(
            f"- Mean gap (E-C): `{dist['entailment_minus_contradiction_mean']:.3f}`"
        )
        lines.append("")
    lines.append("## Stratified Manual Eval Subset")
    lines.append("")
    lines.append(
        f"- Selected rows: `{len(subset_rows)}` "
        f"-> `{output_dir / 'stratified_eval_subset.jsonl'}` and "
        f"`{output_dir / 'manual_annotation_sheet.csv'}`"
    )
    lines.append("- Bucket counts:")
    bucket_counts = Counter(row["bucket"] for row in subset_rows)
    for bucket in BUCKET_ORDER:
        lines.append(f"  - `{bucket}`: {bucket_counts.get(bucket, 0)}")
    lines.append("")
    lines.append("## Label Schema")
    lines.append("")
    lines.append("- Labels: `supported`, `unsupported`, `contradicted`, `partial`, `ambiguous`")
    lines.append("- Error types: " + ", ".join(f"`{item}`" for item in ERROR_TYPES))
    lines.append("")
    return "\n".join(lines)


def compute_summary(
    alias: str,
    config_name: str,
    source_detector_config_name: str,
    isolation: Dict[str, bool],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    detector_available = all(bool(row.get("detector_available", True)) for row in rows)
    detector_errors = sorted(
        {
            str(row.get("hallucination_error")).strip()
            for row in rows
            if str(row.get("hallucination_error") or "").strip()
        }
    )
    detector_model_paths = sorted(
        {
            str(row.get("detector_model_path")).strip()
            for row in rows
            if str(row.get("detector_model_path") or "").strip()
        }
    )
    contradiction_values = [_to_float(row["contradiction_prob_mean"]) for row in rows]
    entailment_values = [_to_float(row["entailment_prob_mean"]) for row in rows]
    neutral_values = [_to_float(row["neutral_prob_mean"]) for row in rows]
    detector_conflicts = [_to_float(row["detector_conflict"]) for row in rows]
    hallucination_topk = [_to_float(row["hallucination_prob_topk"]) for row in rows]
    final_decisions = Counter(row["final_decision"] for row in rows)
    predicted_labels = Counter(row["predicted_detector_label"] for row in rows)
    answered = sum(1 for row in rows if row["final_decision"] == "answer")
    total = len(rows)
    abstain = sum(1 for row in rows if row["is_abstain"])
    contradiction_cases = sum(1 for row in rows if row["predicted_detector_label"] == "contradiction")
    unsupported_cases = sum(
        1 for row in rows
        if row["predicted_detector_label"] in {"neutral", "contradiction"}
    )

    distribution = {
        "contradiction_prob_histogram": make_histogram(contradiction_values),
        "entailment_prob_histogram": make_histogram(entailment_values),
        "neutral_prob_histogram": make_histogram(neutral_values),
        "hallucination_prob_topk_histogram": make_histogram(hallucination_topk),
        "detector_conflict_histogram": make_histogram(detector_conflicts),
        "dominant_contradiction_count": predicted_labels.get("contradiction", 0),
        "detector_conflict_trigger_count": sum(1 for value in detector_conflicts if value > 0.0),
        "entailment_minus_contradiction_mean": _mean(
            e - c for e, c in zip(entailment_values, contradiction_values)
        ),
    }

    return {
        "variant": alias,
        "base_config": config_name,
        "detector_source_config": source_detector_config_name,
        "detector_available": detector_available,
        "detector_error": detector_errors[0] if detector_errors else None,
        "detector_errors": detector_errors,
        "detector_model_path": detector_model_paths[0] if detector_model_paths else None,
        "total_questions": total,
        "answered_rate": (answered / total) if total else 0.0,
        "abstain_rate": (abstain / total) if total else 0.0,
        "contradiction_rate": (contradiction_cases / total) if total else 0.0,
        "unsupported_answer_rate": (unsupported_cases / total) if total else 0.0,
        "contradiction_prob_mean_global": _mean(contradiction_values),
        "entailment_prob_mean_global": _mean(entailment_values),
        "neutral_prob_mean_global": _mean(neutral_values),
        "final_decision_counts": dict(final_decisions),
        "predicted_label_counts": dict(predicted_labels),
        "isolation": isolation,
        "distribution_analysis": distribution,
    }


def run_variant(
    alias: str,
    variant_config_name: str,
    base_config_name: str,
    questions: List[Dict[str, Any]],
    output_dir: Path,
    seed: int,
    top_k_contexts: int,
    keep_source_gating: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config, isolation = build_variant_config(
        base_config_name=base_config_name,
        detector_config_name=variant_config_name,
        keep_source_gating=keep_source_gating,
    )
    detector_model_path = resolve_detector_model_path(config)
    if detector_model_path is None:
        raise SystemExit(
            f"Detector model_path is missing for variant '{alias}' "
            f"(source config: {variant_config_name})"
        )
    if not detector_model_path.exists():
        raise SystemExit(
            f"Detector checkpoint not found for variant '{alias}': {detector_model_path} "
            f"(source config: {variant_config_name})"
        )
    rag = RAGPipeline.from_config(config)
    if rag.vector_store.get_count() == 0:
        raise SystemExit(
            f"Vector collection is empty for variant '{alias}' "
            f"(source config: {variant_config_name})"
        )

    abstain_message = str((config.get("gating") or {}).get("abstain_message", "")).strip()
    rows: List[Dict[str, Any]] = []
    for item in questions:
        result = rag.query(
            query_text=item.get("query", ""),
            return_context=True,
            return_sources=True,
            detect_hallucinations=True,
        )
        gating = result.get("gating") or {}
        stats = gating.get("stats") or {}
        contexts = result.get("context") or []
        generated_answer = (result.get("answer") or "").strip()
        final_decision = "abstain" if (abstain_message and generated_answer == abstain_message) else "answer"
        row = {
            "id": f"{alias}::{item.get('id')}",
            "detector_variant": alias,
            "question_id": item.get("id"),
            "question_type": item.get("type", "unknown"),
            "question": item.get("query", ""),
            "generated_answer": generated_answer,
            "retrieved_chunks": extract_retrieved_chunks(contexts, top_k_contexts),
            "retrieved_chunk_count": len(contexts),
            "retrieval_max_score": _to_float(stats.get("retrieval_max_score")),
            "retrieval_mean_score": _to_float(stats.get("retrieval_mean_score")),
            "source_consistency": _to_float(stats.get("source_consistency")),
            "contradiction_prob_mean": _to_float(stats.get("contradiction_prob_mean")),
            "entailment_prob_mean": _to_float(stats.get("entailment_prob_mean")),
            "neutral_prob_mean": _to_float(stats.get("neutral_prob_mean")),
            "hallucination_prob_topk": _to_float(stats.get("hallucination_prob_topk")),
            "detector_conflict": _to_float(stats.get("detector_conflict")),
            "hard_contradiction_rate": _to_float(stats.get("hard_contradiction_rate")),
            "hallucination_score": _to_float(result.get("hallucination_score")),
            "hallucination_detected": result.get("hallucination_detected"),
            "hallucination_error": result.get("hallucination_error"),
            "detector_available": bool(result.get("hallucination_detected") is not None),
            "detector_model_path": str(detector_model_path),
            "predicted_detector_label": "",
            "final_decision": final_decision,
            "gate_action": gating.get("action", "none"),
            "is_abstain": bool(final_decision == "abstain"),
            "seed": seed,
            "detector_source_config": variant_config_name,
            "base_config": base_config_name,
        }
        row["predicted_detector_label"] = dominant_detector_label(row)
        rows.append(row)

    try:
        import torch  # type: ignore

        del rag
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    summary = compute_summary(
        alias=alias,
        config_name=base_config_name,
        source_detector_config_name=variant_config_name,
        isolation=isolation,
        rows=rows,
    )
    variant_dir = output_dir / alias
    dump_jsonl(variant_dir / "per_question.jsonl", rows)
    dump_json(variant_dir / "summary.json", summary)
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated finreg detector diagnostics")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--questions", default=DEFAULT_QUESTION_SET)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Alias or alias=config_name. Defaults: fever_local, balanced, targeted_contraguard",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k-contexts", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=60)
    parser.add_argument(
        "--subset-variant",
        action="append",
        default=[],
        help="Limit subset generation to specific detector aliases.",
    )
    parser.add_argument("--suspicious-conflict-threshold", type=float, default=0.08)
    parser.add_argument("--suspicious-hallucination-topk-threshold", type=float, default=0.10)
    parser.add_argument("--contradiction-signal-threshold", type=float, default=0.10)
    parser.add_argument("--high-contradiction-gap-threshold", type=float, default=0.50)
    parser.add_argument("--uncertainty-gap-threshold", type=float, default=0.08)
    parser.add_argument("--high-entailment-threshold", type=float, default=0.55)
    parser.add_argument("--low-contradiction-threshold", type=float, default=0.05)
    parser.add_argument(
        "--use-source-gating",
        action="store_true",
        help="Use gating from each source config instead of keeping base gating fixed.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/finreg_detector_diagnostic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(Path(args.questions))
    if args.limit:
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.limit]

    variants = resolve_variants(args.variant)
    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}

    for alias, config_name in variants:
        rows, summary = run_variant(
            alias=alias,
            variant_config_name=config_name,
            base_config_name=args.base_config,
            questions=questions,
            output_dir=output_dir,
            seed=args.seed,
            top_k_contexts=args.top_k_contexts,
            keep_source_gating=bool(args.use_source_gating),
        )
        all_rows.extend(rows)
        summaries[alias] = summary

    valid_rows = [row for row in all_rows if row.get("detector_available", True)]
    if args.subset_variant:
        subset_aliases = {item.strip() for item in args.subset_variant if item.strip()}
        valid_rows = [
            row for row in valid_rows
            if str(row.get("detector_variant", "")).strip() in subset_aliases
        ]
    subset_rows = sample_stratified_subset(
        records=valid_rows,
        total_target=args.subset_size,
        seed=args.seed,
        suspicious_conflict_threshold=args.suspicious_conflict_threshold,
        suspicious_hallucination_topk_threshold=args.suspicious_hallucination_topk_threshold,
        contradiction_signal_threshold=args.contradiction_signal_threshold,
        high_contradiction_gap_threshold=args.high_contradiction_gap_threshold,
        uncertainty_gap_threshold=args.uncertainty_gap_threshold,
        high_entailment_threshold=args.high_entailment_threshold,
        low_contradiction_threshold=args.low_contradiction_threshold,
    )
    dump_jsonl(output_dir / "stratified_eval_subset.jsonl", subset_rows)

    annotation_rows = build_annotation_rows(
        subset=subset_rows,
        top_k_contexts=args.top_k_contexts,
    )
    write_annotation_csv(output_dir / "manual_annotation_sheet.csv", annotation_rows)

    per_detector_summary = {
        "base_config": args.base_config,
        "questions": args.questions,
        "seed": args.seed,
        "variants": summaries,
    }
    dump_json(output_dir / "per_detector_summary.json", per_detector_summary)

    report = build_markdown_report(
        output_dir=output_dir,
        base_config_name=args.base_config,
        question_path=Path(args.questions),
        variants=variants,
        summaries=summaries,
        subset_rows=subset_rows,
    )
    (output_dir / "detector_comparison_report.md").write_text(report, encoding="utf-8")

    dump_json(
        output_dir / "annotation_schema.json",
        {
            "labels": [
                "supported",
                "unsupported",
                "contradicted",
                "partial",
                "ambiguous",
            ],
            "error_types": list(ERROR_TYPES),
        },
    )

    print(f"Wrote outputs to {output_dir}")
    print(f"- {output_dir / 'detector_comparison_report.md'}")
    print(f"- {output_dir / 'per_detector_summary.json'}")
    print(f"- {output_dir / 'stratified_eval_subset.jsonl'}")
    print(f"- {output_dir / 'manual_annotation_sheet.csv'}")


if __name__ == "__main__":
    main()
