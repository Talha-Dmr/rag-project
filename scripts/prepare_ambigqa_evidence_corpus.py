#!/usr/bin/env python3
"""
Prepare a Wikipedia-derived corpus from AmbigQA evidence articles.

Input: AmbigNQ evidence JSON (train/dev/test) where each item contains
- id, question, annotations, articles_plain_text (list of wiki articles)

Output: JSONL with fields: content, metadata
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile
from hashlib import md5
from typing import Dict, Iterable, Iterator, List, Optional, TextIO


def _stream_json_array(fp: TextIO) -> Iterator[Dict]:
    """Yield objects from a top-level JSON array without loading everything."""
    decoder = json.JSONDecoder()
    buf = ""
    in_array = False

    while True:
        chunk = fp.read(1024 * 1024)
        if not chunk:
            break
        buf += chunk

        while True:
            buf = buf.lstrip()
            if not buf:
                break
            if not in_array:
                if buf[0] != '[':
                    raise ValueError("Expected JSON array")
                buf = buf[1:]
                in_array = True
                continue
            if buf[0] == ']':
                return
            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            yield obj
            buf = buf[idx:]
            buf = buf.lstrip()
            if buf.startswith(','):
                buf = buf[1:]
                continue
            if buf.startswith(']'):
                return

    # Drain any remaining buffer
    buf = buf.lstrip()
    if buf and buf != ']':
        try:
            obj, _ = decoder.raw_decode(buf)
            yield obj
        except json.JSONDecodeError:
            pass


def _open_input(path: str, member: Optional[str]) -> TextIO:
    if path.endswith('.zip'):
        if not member:
            raise ValueError("--member is required when input is a .zip")
        zf = zipfile.ZipFile(path)
        if member not in zf.namelist():
            raise FileNotFoundError(f"Member {member} not found in {path}")
        return io.TextIOWrapper(zf.open(member, 'r'), encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def _extract_title(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text else ""
    title = first_line.lstrip('#').strip()
    return title or "unknown"


def _chunk_text(text: str, max_chars: int, min_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
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
    member: Optional[str],
    target_docs: int,
    max_articles_per_item: int,
    max_chars: int,
    min_chars: int,
    seed: int,
    dedupe: bool,
) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen = set()
    doc_count = 0
    item_count = 0

    with _open_input(input_path, member) as fp, open(output_path, 'w', encoding='utf-8') as out:
        for item in _stream_json_array(fp):
            item_count += 1
            question_id = item.get('id')
            question = item.get('question')
            articles = item.get('articles_plain_text') or []

            for a_idx, article in enumerate(articles[:max_articles_per_item]):
                if not article:
                    continue
                title = _extract_title(article)
                chunks = _chunk_text(article, max_chars=max_chars, min_chars=min_chars)

                for c_idx, chunk in enumerate(chunks):
                    if dedupe:
                        h = md5(chunk.encode('utf-8')).hexdigest()
                        if h in seen:
                            continue
                        seen.add(h)

                    record = {
                        "content": chunk,
                        "metadata": {
                            "source": "AmbigQA_evidence",
                            "question_id": question_id,
                            "question": question,
                            "article_title": title,
                            "article_index": a_idx,
                            "chunk_index": c_idx,
                        },
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    doc_count += 1

                    if doc_count % 1000 == 0:
                        print(f"[progress] docs={doc_count} items={item_count}")

                    if doc_count >= target_docs:
                        print(f"Reached target_docs={target_docs}. Stopping.")
                        return doc_count

    return doc_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare corpus from AmbigQA evidence articles")
    parser.add_argument(
        "--input",
        default="data/ambiguity_datasets/02_ambigqa/train_with_evidence_articles.json",
        help="Path to evidence JSON or .zip",
    )
    parser.add_argument(
        "--member",
        default=None,
        help="JSON member inside zip (required if --input is .zip)",
    )
    parser.add_argument(
        "--output",
        default="data/corpora/ambigqa_wiki_evidence.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--target-docs", type=int, default=40000)
    parser.add_argument("--max-articles-per-item", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedupe", action="store_true", help="Disable content deduplication")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        sys.exit(1)

    doc_count = build_corpus(
        input_path=args.input,
        output_path=args.output,
        member=args.member,
        target_docs=args.target_docs,
        max_articles_per_item=args.max_articles_per_item,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        seed=args.seed,
        dedupe=not args.no_dedupe,
    )

    print(f"Done. Wrote {doc_count} docs to {args.output}")


if __name__ == "__main__":
    main()
