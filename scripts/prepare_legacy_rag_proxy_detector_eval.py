#!/usr/bin/env python3
"""Convert legacy RAG QA eval logs into detector proxy inputs.

This is not a gold hallucination benchmark.  It preserves QA reference metrics
so detector predictions can be analyzed against answer correctness signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def context_to_text(selected_context: Any, top_k: int) -> str:
    if isinstance(selected_context, str):
        return selected_context
    if not isinstance(selected_context, list):
        return ""
    chunks: list[str] = []
    for index, item in enumerate(selected_context[:top_k], start=1):
        if isinstance(item, dict):
            chunk_id = item.get("id") or f"ctx_{index}"
            text = str(item.get("text") or item.get("content") or "")
        else:
            chunk_id = f"ctx_{index}"
            text = str(item)
        if text.strip():
            chunks.append(f"[{chunk_id}]\n{text.strip()}")
    return "\n\n".join(chunks)


def build_rows(input_path: Path, *, top_k_contexts: int) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in read_jsonl(input_path):
        architecture = str(row.get("architecture") or input_path.parents[1].name)
        question_id = str(row.get("question_id") or len(output_rows))
        metrics = row.get("metrics") or {}
        reference_answer = row.get("reference_answer")
        output_rows.append(
            {
                "id": f"{architecture}_{question_id}",
                "query": str(row.get("question") or ""),
                "candidate_answer": str(row.get("answer") or ""),
                "evidence_span": context_to_text(row.get("selected_context"), top_k_contexts),
                "reference_answer": reference_answer,
                "qa_exact_match": float(metrics.get("exact_match", 0.0) or 0.0),
                "qa_token_f1": float(metrics.get("token_f1", 0.0) or 0.0),
                "metadata": {
                    "architecture": architecture,
                    "architecture_label": row.get("architecture_label"),
                    "question_id": question_id,
                    "retrieved_doc_ids": row.get("retrieved_doc_ids") or [],
                    "reranked_doc_ids": row.get("reranked_doc_ids") or [],
                    "source_path": str(input_path),
                    "proxy_eval_note": (
                        "Legacy QA reference data, not a support-status gold label."
                    ),
                },
            }
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--top-k-contexts", type=int, default=6)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for input_path in args.input:
        built = build_rows(input_path, top_k_contexts=args.top_k_contexts)
        rows.extend(built)
        source_counts[str(input_path)] = len(built)

    write_jsonl(args.output, rows)
    summary = {
        "output": str(args.output),
        "rows": len(rows),
        "source_counts": source_counts,
        "top_k_contexts": args.top_k_contexts,
        "label_status": "proxy_only_no_support_status_gold",
    }
    write_json(args.summary or args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
