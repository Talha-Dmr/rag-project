#!/usr/bin/env python3
"""
Fetch phase-1 official finreg source documents into the raw corpus tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


DEFAULT_MANIFEST = "config/finreg_phase1_sources.yaml"
DEFAULT_RAW_ROOT = "data/raw/finreg"
DEFAULT_TIMEOUT_SECONDS = 60
USER_AGENT = "rag-project-finreg-corpus-fetcher/1.0"


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"Manifest has no documents: {path}")
    return documents


def should_fetch(entry: Dict[str, Any], wanted_ids: set[str] | None) -> bool:
    if not wanted_ids:
        return True
    return entry["document_id"] in wanted_ids


def fetch_one(url: str, timeout_seconds: int) -> tuple[bytes, Dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        data = response.read()
        headers = {k.lower(): v for k, v in response.headers.items()}
        return data, headers


def save_entry(
    entry: Dict[str, Any],
    raw_root: Path,
    force: bool,
    timeout_seconds: int,
) -> Dict[str, Any]:
    relpath = Path(entry["raw_relpath"])
    target_path = raw_root / relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "document_id": entry["document_id"],
        "source_url": entry["source_url"],
        "download_url": entry["download_url"],
        "raw_path": str(target_path),
        "status": "skipped_existing",
        "fetched_at": None,
        "bytes": None,
        "content_type": None,
        "error": None,
    }

    if target_path.exists() and not force:
        result["bytes"] = target_path.stat().st_size
        return result

    try:
        payload, headers = fetch_one(entry["download_url"], timeout_seconds)
        target_path.write_bytes(payload)
        result["status"] = "fetched"
        result["fetched_at"] = datetime.now(UTC).isoformat()
        result["bytes"] = len(payload)
        result["content_type"] = headers.get("content-type")
        return result
    except (HTTPError, URLError, TimeoutError) as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result


def write_fetch_manifest(raw_root: Path, results: List[Dict[str, Any]]) -> Path:
    out_path = raw_root / "fetch_manifest.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "document_count": len(results),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch phase-1 finreg source documents")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="YAML manifest path")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Raw output root")
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Fetch only specific document_id values; can be repeated.",
    )
    parser.add_argument("--force", action="store_true", help="Refetch even if file exists")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    raw_root = Path(args.raw_root)
    wanted_ids = set(args.document_id or [])

    documents = load_manifest(manifest_path)
    results: List[Dict[str, Any]] = []

    for entry in documents:
        if not should_fetch(entry, wanted_ids):
            continue
        result = save_entry(entry, raw_root, args.force, args.timeout_seconds)
        results.append(result)
        print(
            f"{result['status']:>16}  {entry['document_id']:<28}  "
            f"{result['raw_path']}"
        )
        if result["error"]:
            print(f"  error: {result['error']}")

    manifest_out = write_fetch_manifest(raw_root, results)
    fetched = sum(1 for r in results if r["status"] == "fetched")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped_existing")
    print(
        f"Wrote fetch manifest: {manifest_out} | fetched={fetched} "
        f"skipped={skipped} errors={errors}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
