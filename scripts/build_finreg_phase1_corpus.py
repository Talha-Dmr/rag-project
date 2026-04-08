#!/usr/bin/env python3
"""
Build a processed JSONL corpus from fetched phase-1 finreg source documents.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from pypdf import PdfReader


DEFAULT_MANIFEST = "config/finreg_phase1_sources.yaml"
DEFAULT_RAW_ROOT = "data/raw/finreg"
DEFAULT_PROCESSED_ROOT = "data/processed/finreg"
DEFAULT_OUTPUT = "data/processed/finreg/finreg_phase1_corpus.jsonl"


try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - exercised only in lean environments
    BeautifulSoup = None


@dataclass
class ProcessedDocument:
    content: str
    pages: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignore_depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignore_depth:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignore_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "\n".join(self._parts)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"Manifest has no documents: {path}")
    return documents


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def extract_pdf(raw_path: Path) -> ProcessedDocument:
    reader = PdfReader(str(raw_path))
    pages: List[Dict[str, Any]] = []
    parts: List[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = normalize_whitespace(page.extract_text() or "")
        if not page_text:
            continue
        pages.append({"page": page_num, "text": page_text})
        parts.append(f"[Page {page_num}]\n{page_text}")

    content = "\n\n".join(parts).strip()
    metadata = {"page_count": len(reader.pages), "extracted_page_count": len(pages)}
    return ProcessedDocument(content=content, pages=pages, metadata=metadata)


def extract_html(raw_path: Path) -> ProcessedDocument:
    html = raw_path.read_text(encoding="utf-8", errors="ignore")
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        body = soup.body or soup
        text = normalize_whitespace(body.get_text("\n"))
    else:
        parser = _HTMLTextExtractor()
        parser.feed(html)
        text = normalize_whitespace(parser.text())
    metadata = {"page_count": 1, "extracted_page_count": 1}
    pages = [{"page": 1, "text": text}] if text else []
    return ProcessedDocument(content=text, pages=pages, metadata=metadata)


def extract_document(entry: Dict[str, Any], raw_root: Path) -> ProcessedDocument:
    raw_path = raw_root / entry["raw_relpath"]
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file for {entry['document_id']}: {raw_path}")

    fmt = entry["format"].lower()
    if fmt == "pdf":
        return extract_pdf(raw_path)
    if fmt == "html":
        return extract_html(raw_path)
    raise ValueError(f"Unsupported format for {entry['document_id']}: {fmt}")


def build_record(entry: Dict[str, Any], extracted: ProcessedDocument, raw_root: Path) -> Dict[str, Any]:
    raw_path = raw_root / entry["raw_relpath"]
    return {
        "id": entry["document_id"],
        "content": extracted.content,
        "source_org": entry["source_org"],
        "family": entry["family"],
        "jurisdiction": entry["jurisdiction"],
        "document_type": entry["document_type"],
        "year": entry["year"],
        "title": entry["title"],
        "source_url": entry["source_url"],
        "download_url": entry["download_url"],
        "raw_relpath": entry["raw_relpath"],
        "raw_path": str(raw_path),
        "source_type": "official_phase1_document",
        "themes": entry.get("themes", []),
        **extracted.metadata,
    }


def should_include(entry: Dict[str, Any], wanted_ids: set[str] | None) -> bool:
    if not wanted_ids:
        return True
    return entry["document_id"] in wanted_ids


def write_outputs(
    entry: Dict[str, Any],
    processed: ProcessedDocument,
    record: Dict[str, Any],
    processed_root: Path,
) -> None:
    family_dir = processed_root / entry["family"]
    family_dir.mkdir(parents=True, exist_ok=True)

    stem = family_dir / entry["document_id"]
    stem.with_suffix(".txt").write_text(processed.content, encoding="utf-8")
    stem.with_suffix(".pages.json").write_text(
        json.dumps(processed.pages, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    stem.with_suffix(".metadata.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def write_corpus(records: Iterable[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed finreg phase-1 corpus")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="YAML manifest path")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Raw input root")
    parser.add_argument(
        "--processed-root",
        default=DEFAULT_PROCESSED_ROOT,
        help="Processed output root",
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Output JSONL corpus path")
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Build only specific document_id values; can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    out_path = Path(args.out)
    wanted_ids = set(args.document_id or [])

    records: List[Dict[str, Any]] = []
    failures: List[str] = []

    for entry in manifest:
        if not should_include(entry, wanted_ids):
            continue
        try:
            processed = extract_document(entry, raw_root)
            if not processed.content.strip():
                raise ValueError("empty extracted content")
            record = build_record(entry, processed, raw_root)
            write_outputs(entry, processed, record, processed_root)
            records.append(record)
            print(
                f"{entry['document_id']:<28}  chars={len(processed.content):>7}  "
                f"pages={len(processed.pages):>3}"
            )
        except Exception as exc:  # pragma: no cover - CLI error path
            failures.append(f"{entry['document_id']}: {exc}")
            print(f"error  {entry['document_id']}: {exc}", file=sys.stderr)

    count = write_corpus(records, out_path)
    print(f"Wrote corpus: {out_path} | documents={count} | failures={len(failures)}")
    if failures:
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
