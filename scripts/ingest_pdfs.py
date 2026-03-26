import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import fitz  # PyMuPDF


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Long-term correct structure:
# input  -> data/raw/finreg/<source>/*.pdf
# output -> data/processed/finreg/{texts,chunks,metadata}/<source>/*
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "finreg"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "finreg"

TEXTS_DIR = PROCESSED_DIR / "texts"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
METADATA_DIR = PROCESSED_DIR / "metadata"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_TEXT_THRESHOLD = 300

SUPPORTED_EXTENSIONS = {".pdf"}


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    for d in [TEXTS_DIR, CHUNKS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s/]+", "_", value)
    return value


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def detect_regulator_from_path(pdf_path: Path) -> str:
    parts = [p.lower() for p in pdf_path.parts]

    if "basel" in parts:
        return "BCBS"
    if "iosco" in parts:
        return "IOSCO"
    if "eu" in parts:
        return "EU"
    if "academic" in parts:
        return "ACADEMIC"
    if "uk_boe" in parts or "boe" in parts or "pra" in parts:
        return "PRA_BOE"

    return "UNKNOWN"


def detect_doc_type(regulator: str) -> str:
    mapping = {
        "BCBS": "prudential_standard",
        "IOSCO": "securities_principles",
        "EU": "financial_regulation",
        "ACADEMIC": "academic_reference",
        "PRA_BOE": "prudential_regulation",
        "UNKNOWN": "unknown",
    }
    return mapping.get(regulator, "unknown")


def detect_jurisdiction(regulator: str) -> str:
    mapping = {
        "BCBS": "global",
        "IOSCO": "global",
        "EU": "eu",
        "ACADEMIC": "unknown",
        "PRA_BOE": "uk",
        "UNKNOWN": "unknown",
    }
    return mapping.get(regulator, "unknown")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\xad", "")
    text = re.sub(r"[ \t]+", " ", text)

    # Merge hyphenated line breaks
    text = re.sub(r"(?<!\n)-\n(?=\w)", "", text)

    # Merge broken lines that are probably same paragraph
    text = re.sub(r"(?<![.!?:;\n])\n(?=[a-zA-Z0-9(])", " ", text)

    # Collapse too many newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove isolated page numbers
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Trim spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict]]:
    doc = fitz.open(pdf_path)
    full_text_parts = []
    page_records = []

    for page_index, page in enumerate(doc):
        page_text = page.get_text("text")
        cleaned_page_text = clean_text(page_text)

        page_records.append(
            {
                "page_num": page_index + 1,
                "text": cleaned_page_text,
                "char_count": len(cleaned_page_text),
            }
        )

        if cleaned_page_text:
            full_text_parts.append(f"\n\n[PAGE {page_index + 1}]\n{cleaned_page_text}")

    full_text = "\n".join(full_text_parts).strip()
    return full_text, page_records


def extract_title(full_text: str, fallback_name: str) -> str:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    for line in lines[:15]:
        if 10 < len(line) < 200 and not line.startswith("[PAGE "):
            return line
    return fallback_name


def extract_year_from_filename(filename: str) -> str:
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return match.group(0)
    return "unknown"


def build_output_subdirs(pdf_path: Path) -> Tuple[Path, Path, Path]:
    # Example:
    # raw/finreg/uk_boe/foo.pdf -> texts/uk_boe, chunks/uk_boe, metadata/uk_boe
    relative_parent = pdf_path.parent.relative_to(RAW_DIR)

    text_subdir = TEXTS_DIR / relative_parent
    chunk_subdir = CHUNKS_DIR / relative_parent
    metadata_subdir = METADATA_DIR / relative_parent

    text_subdir.mkdir(parents=True, exist_ok=True)
    chunk_subdir.mkdir(parents=True, exist_ok=True)
    metadata_subdir.mkdir(parents=True, exist_ok=True)

    return text_subdir, chunk_subdir, metadata_subdir


def page_lookup_for_char_offset(page_records: List[Dict], chunk_text: str, full_text: str) -> Tuple[int, int]:
    start_idx = full_text.find(chunk_text)
    if start_idx == -1:
        return -1, -1

    end_idx = start_idx + len(chunk_text)

    page_markers = []
    for match in re.finditer(r"\[PAGE (\d+)\]", full_text):
        page_num = int(match.group(1))
        page_markers.append((match.start(), page_num))

    if not page_markers:
        return -1, -1

    start_page = page_markers[0][1]
    end_page = page_markers[-1][1]

    for i, (marker_pos, page_num) in enumerate(page_markers):
        next_pos = page_markers[i + 1][0] if i + 1 < len(page_markers) else len(full_text)
        if marker_pos <= start_idx < next_pos:
            start_page = page_num
        if marker_pos <= end_idx <= next_pos:
            end_page = page_num
            break

    return start_page, end_page


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if end < text_length:
            last_break = max(
                chunk.rfind("\n\n"),
                chunk.rfind(". "),
                chunk.rfind("; "),
            )
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


def build_document_metadata(pdf_path: Path, full_text: str, page_records: List[Dict]) -> Dict:
    regulator = detect_regulator_from_path(pdf_path)
    doc_type = detect_doc_type(regulator)
    jurisdiction = detect_jurisdiction(regulator)

    return {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "source_collection": str(pdf_path.parent.relative_to(RAW_DIR)).replace("\\", "/"),
        "title": extract_title(full_text, pdf_path.stem),
        "regulator": regulator,
        "doc_type": doc_type,
        "jurisdiction": jurisdiction,
        "language": "en",
        "year": extract_year_from_filename(pdf_path.name),
        "num_pages": len(page_records),
        "num_characters": len(full_text),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "needs_ocr_review": len(full_text.strip()) < MIN_TEXT_THRESHOLD,
    }


def build_chunk_records(
    pdf_path: Path,
    full_text: str,
    page_records: List[Dict],
    document_metadata: Dict,
) -> List[Dict]:
    base_id = slugify(pdf_path.stem)
    chunks = chunk_text(full_text)

    chunk_records = []

    for idx, chunk in enumerate(chunks, start=1):
        page_start, page_end = page_lookup_for_char_offset(page_records, chunk, full_text)

        chunk_records.append(
            {
                "chunk_id": f"{base_id}_{idx:04d}",
                "text": chunk,
                "text_sha1": sha1_text(chunk),
                "chunk_index": idx,
                "source_file": document_metadata["source_file"],
                "source_collection": document_metadata["source_collection"],
                "title": document_metadata["title"],
                "regulator": document_metadata["regulator"],
                "doc_type": document_metadata["doc_type"],
                "jurisdiction": document_metadata["jurisdiction"],
                "language": document_metadata["language"],
                "year": document_metadata["year"],
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    return chunk_records


def save_outputs(
    pdf_path: Path,
    full_text: str,
    page_records: List[Dict],
    document_metadata: Dict,
    chunk_records: List[Dict],
) -> None:
    text_subdir, chunk_subdir, metadata_subdir = build_output_subdirs(pdf_path)
    base_name = slugify(pdf_path.stem)

    text_file = text_subdir / f"{base_name}.txt"
    metadata_file = metadata_subdir / f"{base_name}.json"
    pages_file = metadata_subdir / f"{base_name}_pages.json"
    chunks_file = chunk_subdir / f"{base_name}.jsonl"

    text_file.write_text(full_text, encoding="utf-8")
    metadata_file.write_text(
        json.dumps(document_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    pages_file.write_text(
        json.dumps(page_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    with chunks_file.open("w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_pdf_files(root: Path) -> List[Path]:
    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


def process_pdf(pdf_path: Path) -> Dict:
    print(f"[INFO] Processing: {pdf_path}")

    full_text, page_records = extract_text_from_pdf(pdf_path)
    document_metadata = build_document_metadata(pdf_path, full_text, page_records)
    chunk_records = build_chunk_records(pdf_path, full_text, page_records, document_metadata)

    save_outputs(pdf_path, full_text, page_records, document_metadata, chunk_records)

    result = {
        "file": str(pdf_path),
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


def save_manifest(results: List[Dict]) -> None:
    manifest_path = PROCESSED_DIR / "manifest.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(RAW_DIR),
        "processed_root": str(PROCESSED_DIR),
        "total_files": len(results),
        "files": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    pdf_files = iter_pdf_files(RAW_DIR)
    if not pdf_files:
        print("[WARN] No PDF files found.")
        return

    results = []
    for pdf_path in pdf_files:
        try:
            result = process_pdf(pdf_path)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed on {pdf_path}: {e}")

    save_manifest(results)
    print(f"[INFO] Finished. Processed {len(results)} PDF files.")


if __name__ == "__main__":
    main()