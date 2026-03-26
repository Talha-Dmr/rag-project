#!/usr/bin/env python3
"""
Per-example error analysis for hallucination detector checkpoints.

Produces:
- overall metrics
- confusion buckets grouped by true/pred labels
- top high-confidence mistakes
- optional JSONL dump with raw predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_loader import load_config
from src.training.base_trainer import TrainerFactory
from src.training.trainers import hallucination_trainer  # noqa: F401
from src.training.data.nli_dataset import create_dataloader
from src.training.metrics.nli_metrics import NLIMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_errors(records: list[dict[str, Any]], top_k: int = 25) -> dict[str, Any]:
    confusion = Counter()
    error_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in records:
        true_label = row["true_label_name"]
        pred_label = row["pred_label_name"]
        confusion[f"{true_label}->{pred_label}"] += 1
        if row["is_error"]:
            error_buckets[f"{true_label}->{pred_label}"].append(row)

    top_errors = sorted(
        [row for row in records if row["is_error"]],
        key=lambda r: (r["confidence"], r["margin"]),
        reverse=True,
    )[:top_k]

    bucket_summaries = {}
    for key, rows in error_buckets.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (r["confidence"], r["margin"]),
            reverse=True,
        )
        bucket_summaries[key] = {
            "count": len(rows),
            "top_examples": rows_sorted[:5],
        }

    return {
        "confusion_pairs": dict(confusion),
        "error_buckets": bucket_summaries,
        "top_high_confidence_errors": top_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hallucination detector errors")
    parser.add_argument("--model-path", required=True, help="Checkpoint directory")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test jsonl")
    parser.add_argument("--config", required=True, help="Training config name")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to analyze",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--dump-jsonl",
        default="",
        help="Optional per-example JSONL dump path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Override analysis batch size",
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=25,
        help="How many top confidence mistakes to keep",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})

    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    if not data_path.exists():
        raise SystemExit(f"Split not found: {data_path}")

    examples = load_examples(data_path)
    logger.info("Loaded %s raw examples from %s", len(examples), data_path)

    trainer = TrainerFactory.create("hallucination", config=training_config)
    trainer.build_model()
    trainer.load_checkpoint(args.model_path)
    trainer.model.eval()

    data_cfg = training_config.get("data", {})
    model_cfg = training_config.get("model", {})

    loader = create_dataloader(
        data_path=str(data_path),
        tokenizer_name=model_cfg.get("base_model"),
        batch_size=args.batch_size,
        max_length=data_cfg.get("max_seq_length", 256),
        shuffle=False,
        cache_dir=model_cfg.get("cache_dir"),
    )

    metrics_tracker = NLIMetrics()
    records: list[dict[str, Any]] = []
    dump_fh = None
    if args.dump_jsonl:
        dump_path = Path(args.dump_jsonl)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_fh = dump_path.open("w", encoding="utf-8")

    offset = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            outputs = trainer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)

            metrics_tracker.update(
                predictions=predictions.cpu().tolist(),
                labels=labels.cpu().tolist(),
                probabilities=probs.cpu().tolist(),
            )

            probs_cpu = probs.cpu()
            preds_cpu = predictions.cpu()
            labels_cpu = labels.cpu()

            for i in range(len(preds_cpu)):
                raw = examples[offset + i]
                pred_idx = int(preds_cpu[i].item())
                true_idx = int(labels_cpu[i].item())
                prob_vec = [float(x) for x in probs_cpu[i].tolist()]
                sorted_probs = sorted(prob_vec, reverse=True)
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                row = {
                    "index": offset + i,
                    "premise": raw.get("premise"),
                    "hypothesis": raw.get("hypothesis"),
                    "metadata": raw.get("metadata", {}),
                    "true_label": true_idx,
                    "true_label_name": LABEL_NAMES[true_idx],
                    "pred_label": pred_idx,
                    "pred_label_name": LABEL_NAMES[pred_idx],
                    "probabilities": {
                        LABEL_NAMES[j]: prob_vec[j] for j in range(3)
                    },
                    "confidence": float(sorted_probs[0]),
                    "margin": float(margin),
                    "is_error": pred_idx != true_idx,
                }
                records.append(row)
                if dump_fh is not None:
                    dump_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            offset += len(preds_cpu)

    if dump_fh is not None:
        dump_fh.close()

    metrics = metrics_tracker.compute()
    summary = {
        "config": args.config,
        "model_path": args.model_path,
        "split": args.split,
        "data_path": str(data_path),
        "total_examples": len(records),
        "metrics": metrics,
    }
    summary.update(summarize_errors(records, top_k=args.top_k_errors))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Wrote error analysis to %s", out_path)
    if args.dump_jsonl:
        logger.info("Wrote per-example dump to %s", args.dump_jsonl)


if __name__ == "__main__":
    main()
