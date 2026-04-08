#!/usr/bin/env python3
"""
Ingest PDF corpora into a processed text/chunk format while preserving the raw folder hierarchy.

Example:
    poetry run python scripts/ingest_pdfs.py \
        --raw-root data/raw/finreg \
        --processed-root data/processed/finreg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "finreg"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "finreg"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MIN_TEXT_THRESHOLD = 300
SUPPORTED_EXTENSIONS = {".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs into cleaned text, per-page metadata, and chunk records.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root folder containing source PDFs.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root folder where processed outputs will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Target chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--min-text-threshold",
        type=int,
        default=DEFAULT_MIN_TEXT_THRESHOLD,
        help="Flag documents below this extracted character count for OCR review.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed artifacts.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s/]+", "_", value)
    return value


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_pdf_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def detect_regulator_from_path(pdf_path: Path) -> str:
    parts = [part.lower() for part in pdf_path.parts]

    if "basel" in parts:
        return "BCBS"
    if "iosco" in parts:
        return "IOSCO"
    if "eu_regulations" in parts:
        return "EU"
    if "us_regulations" in parts:
        return "US"
    if "eba" in parts:
        return "EBA"
    if "boe" in parts:
        return "PRA_BOE"
    if "academic" in parts:
        return "ACADEMIC"
    if "technical_manuals" in parts:
        return "TECHNICAL"
    if "supporting_docs" in parts:
        return "SUPPORTING"

    return "UNKNOWN"


def detect_doc_type(regulator: str) -> str:
    mapping = {
        "BCBS": "prudential_standard",
        "IOSCO": "principles_standard",
        "EU": "binding_regulation",
        "US": "binding_regulation",
        "EBA": "supervisory_guidance",
        "PRA_BOE": "supervisory_guidance",
        "ACADEMIC": "academic_reference",
        "TECHNICAL": "technical_manual",
        "SUPPORTING": "supporting_reference",
        "UNKNOWN": "unknown",
    }
    return mapping.get(regulator, "unknown")


def detect_jurisdiction(regulator: str) -> str:
    mapping = {
        "BCBS": "global",
        "IOSCO": "global",
        "EU": "eu",
        "US": "us",
        "EBA": "eu",
        "PRA_BOE": "uk",
        "ACADEMIC": "unknown",
        "TECHNICAL": "unknown",
        "SUPPORTING": "unknown",
        "UNKNOWN": "unknown",
    }
    return mapping.get(regulator, "unknown")


def detect_layer(pdf_path: Path) -> str:
    parts = [part.lower() for part in pdf_path.parts]
    for part in parts:
        if part.startswith("layer_"):
            return part
    return "unlayered"


def extract_year_from_filename(filename: str) -> str:
    match = re.search(r"(19|20)\d{2}", filename)
    return match.group(0) if match else "unknown"


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\xad", "")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)-\n(?=\w)", "", text)
    text = re.sub(r"(?<![.!?:;\n])\n(?=[a-z0-9(])", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*page\s+\d+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*(confidential|draft|publication version)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    previous_blank = False

    for line in lines:
        if is_junk_line(line):
            continue

        if not line:
            if previous_blank:
                continue
            cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def is_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if len(stripped) <= 2 and not any(char.isalpha() for char in stripped):
        return True

    if re.fullmatch(r"[\d\s./-]+", stripped):
        return True

    if re.fullmatch(r"[A-Z\s]{3,}", stripped) and len(stripped.split()) <= 6:
        return False

    repeated_punct = sum(char in "_-." for char in stripped)
    if repeated_punct >= max(8, int(len(stripped) * 0.7)):
        return True

    return False


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict[str, object]]]:
    full_text_parts: List[str] = []
    page_records: List[Dict[str, object]] = []
    reader = PdfReader(str(pdf_path))

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        cleaned_page_text = clean_text(page_text)

        page_records.append(
            {
                "page_num": page_index,
                "text": cleaned_page_text,
                "char_count": len(cleaned_page_text),
            }
        )

        if cleaned_page_text:
            full_text_parts.append(f"[PAGE {page_index}]\n{cleaned_page_text}")

    full_text = "\n\n".join(full_text_parts).strip()
    return full_text, page_records


def extract_title(full_text: str, fallback_name: str) -> str:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    for line in lines[:20]:
        if line.startswith("[PAGE "):
            continue
        if line.lower().startswith("downloaded on "):
            continue
        if line.lower().startswith("bank for international settlements"):
            continue
        if 10 <= len(line) <= 220:
            return line
    return fallback_name


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if end < text_length:
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("; "))
            if last_break > int(chunk_size * 0.6):
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


def normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("[PAGE "):
        return False
    if len(stripped) > 120:
        return False
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
        return True
    if stripped.isupper() and any(char.isalpha() for char in stripped):
        return True
    return False


def structure_aware_segments(full_text: str, prefer_headings: bool) -> List[str]:
    segments: List[str] = []
    current_lines: List[str] = []

    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                segments.append("\n".join(current_lines).strip())
                current_lines = []
            continue

        if line.startswith("[PAGE "):
            if current_lines:
                segments.append("\n".join(current_lines).strip())
                current_lines = []
            continue

        if prefer_headings and is_heading_line(line) and current_lines:
            segments.append("\n".join(current_lines).strip())
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        segments.append("\n".join(current_lines).strip())

    return [segment for segment in segments if segment]


def pack_segments(segments: Sequence[str], chunk_size: int, overlap: int) -> List[str]:
    if not segments:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for segment in segments:
        segment_len = len(segment)
        separator_len = 2 if current else 0
        would_exceed = current and current_len + separator_len + segment_len > chunk_size
        if would_exceed:
            chunks.append("\n\n".join(current).strip())

            if overlap > 0:
                overlap_segments: List[str] = []
                overlap_len = 0
                for existing in reversed(current):
                    extra = len(existing) + (2 if overlap_segments else 0)
                    if overlap_segments and overlap_len + extra > overlap:
                        break
                    overlap_segments.insert(0, existing)
                    overlap_len += extra
                current = overlap_segments[:]
                current_len = sum(len(item) for item in current) + max(0, len(current) - 1) * 2
            else:
                current = []
                current_len = 0

        separator_len = 2 if current else 0
        current.append(segment)
        current_len += separator_len + segment_len

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def build_chunk_records_for_document(
    doc_stem: str,
    full_text: str,
    document_metadata: Dict[str, object],
    chunk_strategy: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, object]]:
    strategy = chunk_strategy.lower().strip()
    if strategy == "structure_aware":
        chunks = pack_segments(
            structure_aware_segments(full_text, prefer_headings=False),
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif strategy == "structure_aware_small":
        adjusted_chunk_size = max(400, chunk_size // 2)
        adjusted_overlap = min(overlap, adjusted_chunk_size // 4)
        chunks = pack_segments(
            structure_aware_segments(full_text, prefer_headings=False),
            chunk_size=adjusted_chunk_size,
            overlap=adjusted_overlap,
        )
    elif strategy == "structure_aware_small_heading":
        adjusted_chunk_size = max(400, chunk_size // 2)
        adjusted_overlap = min(overlap, adjusted_chunk_size // 4)
        chunks = pack_segments(
            structure_aware_segments(full_text, prefer_headings=True),
            chunk_size=adjusted_chunk_size,
            overlap=adjusted_overlap,
        )
    else:
        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    base_id = slugify(doc_stem)
    chunk_records: List[Dict[str, object]] = []
    for idx, chunk_value in enumerate(chunks, start=1):
        page_start, page_end = page_lookup_for_chunk(chunk_value, full_text)
        chunk_records.append(
            {
                "chunk_id": f"{base_id}_{idx:04d}",
                "chunk_index": idx,
                "text": chunk_value,
                "text_sha1": sha1_text(chunk_value),
                "source_file": document_metadata["source_file"],
                "source_collection": document_metadata["source_collection"],
                "layer": document_metadata["layer"],
                "title": document_metadata["title"],
                "regulator": document_metadata["regulator"],
                "doc_type": document_metadata["doc_type"],
                "jurisdiction": document_metadata["jurisdiction"],
                "language": document_metadata["language"],
                "year": document_metadata["year"],
                "page_start": page_start,
                "page_end": page_end,
                "chunking_strategy": strategy,
            }
        )
    return chunk_records


def page_lookup_for_chunk(chunk_text_value: str, full_text: str) -> Tuple[int, int]:
    start_idx = full_text.find(chunk_text_value)
    if start_idx == -1:
        return -1, -1

    end_idx = start_idx + len(chunk_text_value)
    page_markers = [(match.start(), int(match.group(1))) for match in re.finditer(r"\[PAGE (\d+)\]", full_text)]

    if not page_markers:
        return -1, -1

    start_page = page_markers[0][1]
    end_page = page_markers[-1][1]

    for index, (marker_pos, page_num) in enumerate(page_markers):
        next_pos = page_markers[index + 1][0] if index + 1 < len(page_markers) else len(full_text)
        if marker_pos <= start_idx < next_pos:
            start_page = page_num
        if marker_pos <= end_idx <= next_pos:
            end_page = page_num
            break

    return start_page, end_page


def build_document_metadata(
    pdf_path: Path,
    raw_root: Path,
    full_text: str,
    page_records: List[Dict[str, object]],
    min_text_threshold: int,
) -> Dict[str, object]:
    regulator = detect_regulator_from_path(pdf_path)
    relative_parent = pdf_path.parent.relative_to(raw_root)
    source_collection = str(relative_parent).replace("\\", "/")

    return {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "source_collection": source_collection,
        "layer": detect_layer(pdf_path),
        "title": extract_title(full_text, pdf_path.stem),
        "regulator": regulator,
        "doc_type": detect_doc_type(regulator),
        "jurisdiction": detect_jurisdiction(regulator),
        "language": "en",
        "year": extract_year_from_filename(pdf_path.name),
        "num_pages": len(page_records),
        "num_characters": len(full_text),
        "text_sha1": sha1_text(full_text) if full_text else None,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "needs_ocr_review": len(full_text.strip()) < min_text_threshold,
    }


def build_chunk_records(
    pdf_path: Path,
    full_text: str,
    document_metadata: Dict[str, object],
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, object]]:
    return build_chunk_records_for_document(
        doc_stem=pdf_path.stem,
        full_text=full_text,
        document_metadata=document_metadata,
        chunk_strategy="fixed",
        chunk_size=chunk_size,
        overlap=overlap,
    )


def build_output_paths(pdf_path: Path, raw_root: Path, processed_root: Path) -> Dict[str, Path]:
    relative_parent = pdf_path.parent.relative_to(raw_root)
    output_dir = processed_root / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = slugify(pdf_path.stem)
    return {
        "dir": output_dir,
        "text": output_dir / f"{base_name}.txt",
        "metadata": output_dir / f"{base_name}.metadata.json",
        "pages": output_dir / f"{base_name}.pages.json",
        "chunks": output_dir / f"{base_name}.chunks.jsonl",
    }


def outputs_exist(output_paths: Dict[str, Path]) -> bool:
    return all(output_paths[key].exists() for key in ("text", "metadata", "pages", "chunks"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_pdf(
    pdf_path: Path,
    raw_root: Path,
    processed_root: Path,
    chunk_size: int,
    overlap: int,
    min_text_threshold: int,
    force: bool,
) -> Dict[str, object]:
    output_paths = build_output_paths(pdf_path, raw_root, processed_root)
    if outputs_exist(output_paths) and not force:
        return {
            "file": str(pdf_path),
            "status": "skipped_existing",
            "source_collection": str(pdf_path.parent.relative_to(raw_root)).replace("\\", "/"),
        }

    print(f"[INFO] Processing: {pdf_path}")
    full_text, page_records = extract_text_from_pdf(pdf_path)
    document_metadata = build_document_metadata(
        pdf_path=pdf_path,
        raw_root=raw_root,
        full_text=full_text,
        page_records=page_records,
        min_text_threshold=min_text_threshold,
    )
    chunk_records = build_chunk_records(
        pdf_path=pdf_path,
        full_text=full_text,
        document_metadata=document_metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    output_paths["text"].write_text(full_text, encoding="utf-8")
    write_json(output_paths["metadata"], document_metadata)
    write_json(output_paths["pages"], page_records)
    write_jsonl(output_paths["chunks"], chunk_records)

    result = {
        "file": str(pdf_path),
        "status": "processed",
        "source_collection": document_metadata["source_collection"],
        "title": document_metadata["title"],
        "pages": document_metadata["num_pages"],
        "chars": document_metadata["num_characters"],
        "chunks": len(chunk_records),
        "needs_ocr_review": document_metadata["needs_ocr_review"],
    }

    print(
        f"[DONE] {pdf_path.name} | source={result['source_collection']} "
        f"| pages={result['pages']} | chars={result['chars']} "
        f"| chunks={result['chunks']} | ocr_review={result['needs_ocr_review']}"
    )
    return result


def save_manifest(
    processed_root: Path,
    raw_root: Path,
    results: List[Dict[str, object]],
    chunk_size: int,
    overlap: int,
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(raw_root),
        "processed_root": str(processed_root),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "total_files": len(results),
        "processed_files": sum(result.get("status") == "processed" for result in results),
        "skipped_existing_files": sum(result.get("status") == "skipped_existing" for result in results),
        "files": results,
    }
    write_json(processed_root / "manifest.json", manifest)


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if args.chunk_overlap < 0:
        raise SystemExit("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be smaller than --chunk-size")
    if not args.raw_root.exists():
        raise SystemExit(f"Raw root not found: {args.raw_root}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.processed_root.mkdir(parents=True, exist_ok=True)

    pdf_files = iter_pdf_files(args.raw_root)
    if not pdf_files:
        print("[WARN] No PDF files found.")
        return

    results: List[Dict[str, object]] = []
    for pdf_path in pdf_files:
        try:
            result = process_pdf(
                pdf_path=pdf_path,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                min_text_threshold=args.min_text_threshold,
                force=args.force,
            )
            results.append(result)
        except Exception as exc:
            error_result = {
                "file": str(pdf_path),
                "status": "error",
                "error": str(exc),
            }
            results.append(error_result)
            print(f"[ERROR] Failed on {pdf_path}: {exc}")

    save_manifest(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        results=results,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    print(
        f"[INFO] Finished. total={len(results)} "
        f"processed={sum(result.get('status') == 'processed' for result in results)} "
        f"skipped={sum(result.get('status') == 'skipped_existing' for result in results)} "
        f"errors={sum(result.get('status') == 'error' for result in results)}"
    )


if __name__ == "__main__":
    main()
