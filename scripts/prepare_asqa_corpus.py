#!/usr/bin/env python3
"""Build a JSONL corpus from ASQA knowledge passages."""

import argparse
import json
import os
from hashlib import md5
from typing import Dict, List


def chunk_text(text: str, max_chars: int, min_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 > max_chars and current:
            chunk = "\n\n".join(current).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunk = "\n\n".join(current).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)

    return chunks


def build_corpus(
    input_path: str,
    output_path: str,
    target_docs: int,
    max_chars: int,
    min_chars: int,
    include_qa_pairs: bool,
    dedupe: bool,
) -> int:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen = set()
    doc_count = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        for split_name, split_data in data.items():
            for ex_id, ex in split_data.items():
                ambiguous_question = ex.get('ambiguous_question')

                for ann in ex.get('annotations', []):
                    for k in ann.get('knowledge', []):
                        content = k.get('content')
                        if not content:
                            continue
                        wikipage = k.get('wikipage')
                        chunks = chunk_text(content, max_chars=max_chars, min_chars=min_chars)
                        for c_idx, chunk in enumerate(chunks):
                            if dedupe:
                                h = md5(chunk.encode('utf-8')).hexdigest()
                                if h in seen:
                                    continue
                                seen.add(h)
                            record = {
                                "content": chunk,
                                "metadata": {
                                    "source": "ASQA",
                                    "split": split_name,
                                    "example_id": ex_id,
                                    "ambiguous_question": ambiguous_question,
                                    "wikipage": wikipage,
                                    "chunk_index": c_idx,
                                },
                            }
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            doc_count += 1
                            if doc_count % 1000 == 0:
                                print(f"[progress] docs={doc_count}")
                            if doc_count >= target_docs:
                                print(f"Reached target_docs={target_docs}. Stopping.")
                                return doc_count

                if include_qa_pairs:
                    for qa in ex.get('qa_pairs', []):
                        q = qa.get('question')
                        answers = qa.get('short_answers') or []
                        content = f"Question: {q}\nAnswer: {', '.join(answers)}"
                        if dedupe:
                            h = md5(content.encode('utf-8')).hexdigest()
                            if h in seen:
                                continue
                            seen.add(h)
                        record = {
                            "content": content,
                            "metadata": {
                                "source": "ASQA_qa_pairs",
                                "split": split_name,
                                "example_id": ex_id,
                                "ambiguous_question": ambiguous_question,
                            },
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        doc_count += 1
                        if doc_count % 1000 == 0:
                            print(f"[progress] docs={doc_count}")
                        if doc_count >= target_docs:
                            print(f"Reached target_docs={target_docs}. Stopping.")
                            return doc_count

    return doc_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASQA corpus JSONL")
    parser.add_argument(
        "--input",
        default="data/ambiguity_datasets/03_asqa/dataset/ASQA.json",
        help="Path to ASQA.json",
    )
    parser.add_argument(
        "--output",
        default="data/corpora/asqa_wiki_knowledge.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--target-docs", type=int, default=20000)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--include-qa-pairs", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    count = build_corpus(
        input_path=args.input,
        output_path=args.output,
        target_docs=args.target_docs,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        include_qa_pairs=args.include_qa_pairs,
        dedupe=not args.no_dedupe,
    )

    print(f"Done. Wrote {count} docs to {args.output}")


if __name__ == "__main__":
    main()
