#!/usr/bin/env python3
"""
Download Hugging Face `pietrolesci/nli_fever` and export it into the local
raw format expected by `FeverConverter`.

Output files:
- data/fever/pair_nli/train.jsonl
- data/fever/pair_nli/val.jsonl
- data/fever/pair_nli/test.jsonl

The exported schema is:
{
  "id": "...",
  "claim": "...",
  "label": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
  "evidence_text": "..."
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SPLIT_URLS = {
    "train": "https://huggingface.co/datasets/pietrolesci/nli_fever/resolve/main/data/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/pietrolesci/nli_fever/resolve/main/data/dev-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/pietrolesci/nli_fever/resolve/main/data/test-00000-of-00001.parquet",
}

ID_TO_LABEL = {
    0: "SUPPORTS",
    1: "NOT ENOUGH INFO",
    2: "REFUTES",
}


def convert_split(split: str, url: str, output_path: Path) -> None:
    df = pd.read_parquet(url)

    required_cols = {"premise", "hypothesis"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{split}: missing required columns: {sorted(missing)}")

    label_col = "fever_gold_label" if "fever_gold_label" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"{split}: no label column found")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            raw_label = row_dict.get(label_col)
            if raw_label is None:
                # The public nli_fever test split may not carry labels.
                continue

            if isinstance(raw_label, (int, float)):
                mapped_label = ID_TO_LABEL[int(raw_label)]
            else:
                mapped_label = str(raw_label).strip()

            record = {
                "id": row_dict.get("fid", row_dict.get("cid", "")),
                "claim": str(row_dict["premise"]).strip(),
                "label": mapped_label,
                "evidence_text": str(row_dict["hypothesis"]).strip(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize nli_fever pair data")
    parser.add_argument(
        "--output-dir",
        default="data/fever/pair_nli",
        help="Directory where train/val/test JSONL files will be written",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for split, url in SPLIT_URLS.items():
        output_path = output_dir / f"{split}.jsonl"
        print(f"Downloading {split} from {url}")
        convert_split(split, url, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
