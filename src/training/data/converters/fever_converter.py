"""
Converter for FEVER-style evidence verification data to NLI format.

This converter is intentionally pair-NLI oriented for the current
hallucination detector task:
  premise = evidence text
  hypothesis = claim text
  label = entailment / neutral / contradiction

Expected raw input:
- A directory containing one or more JSON/JSONL files such as
  train.jsonl, val.jsonl, dev.jsonl, test.jsonl
- Or a single JSON/JSONL file
- Or locally converted JSONL exported from Hugging Face `pietrolesci/nli_fever`

Supported record fields:
- claim / hypothesis
- label
- evidence_text / premise / context / evidence

The evidence field may be:
- a single string
- a list of strings
- a dict containing a text-like field
- a nested list/dict structure containing text-like fields
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.training.data.base_converter import BaseConverter

logger = logging.getLogger(__name__)


class FeverConverter(BaseConverter):
    """Convert FEVER-style local data to pair-NLI format."""

    LABEL_MAP = {
        "supports": BaseConverter.LABEL_ENTAILMENT,
        "entailment": BaseConverter.LABEL_ENTAILMENT,
        "refutes": BaseConverter.LABEL_CONTRADICTION,
        "contradiction": BaseConverter.LABEL_CONTRADICTION,
        "not enough info": BaseConverter.LABEL_NEUTRAL,
        "nei": BaseConverter.LABEL_NEUTRAL,
        "neutral": BaseConverter.LABEL_NEUTRAL,
        "0": BaseConverter.LABEL_ENTAILMENT,
        "1": BaseConverter.LABEL_NEUTRAL,
        "2": BaseConverter.LABEL_CONTRADICTION,
    }

    TEXT_KEYS = (
        "text",
        "sentence",
        "evidence_text",
        "evidence",
        "context",
        "premise",
    )

    def __init__(self, dataset_path: str, multiplier: int = 1, seed: int = 42):
        super().__init__(dataset_path, multiplier, seed)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load FEVER-style raw examples from JSON or JSONL files."""
        file_paths = self._resolve_input_files()
        all_examples: List[Dict[str, Any]] = []

        for file_path in file_paths:
            suffix = file_path.suffix.lower()
            if suffix == ".jsonl":
                loaded = self._load_jsonl(file_path)
            elif suffix == ".json":
                loaded = self._load_json(file_path)
            else:
                logger.warning("Skipping unsupported FEVER file: %s", file_path)
                continue

            all_examples.extend(loaded)
            logger.info("Loaded %s FEVER examples from %s", len(loaded), file_path.name)

        return all_examples

    def convert_to_nli(self, raw_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert FEVER-style raw examples into pair-NLI records."""
        nli_examples: List[Dict[str, Any]] = []

        for raw_ex in raw_examples:
            claim = self._extract_claim(raw_ex)
            if not claim:
                continue

            mapped_label = self._map_label(raw_ex.get("fever_gold_label", raw_ex.get("label")))
            if mapped_label is None:
                continue

            evidence_texts = self._extract_evidence_texts(raw_ex)
            if not evidence_texts:
                logger.debug("Skipping FEVER example without usable evidence text")
                continue

            example_id = raw_ex.get("id", raw_ex.get("example_id", "unknown"))
            source_label = str(raw_ex.get("label", "")).strip()

            for idx, evidence_text in enumerate(evidence_texts):
                evidence_text = evidence_text.strip()
                if not evidence_text:
                    continue

                nli_examples.append(
                    {
                        "premise": evidence_text,
                        "hypothesis": claim,
                        "label": mapped_label,
                        "metadata": {
                            "example_id": example_id,
                            "source_label": source_label,
                            "evidence_index": idx,
                            "source_format": "fever",
                        },
                    }
                )

        return nli_examples

    def _resolve_input_files(self) -> List[Path]:
        if self.dataset_path.is_file():
            return [self.dataset_path]

        candidate_names = (
            "train.jsonl",
            "train.json",
            "val.jsonl",
            "val.json",
            "dev.jsonl",
            "dev.json",
            "validation.jsonl",
            "validation.json",
            "test.jsonl",
            "test.json",
        )

        discovered = [self.dataset_path / name for name in candidate_names if (self.dataset_path / name).exists()]
        if discovered:
            return discovered

        discovered = sorted(
            [
                path
                for path in self.dataset_path.iterdir()
                if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
            ]
        )
        if discovered:
            return discovered

        raise FileNotFoundError(f"No FEVER JSON/JSONL files found under {self.dataset_path}")

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        loaded: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                loaded.append(json.loads(line))
        return loaded

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported FEVER JSON structure in {file_path}")

    def _map_label(self, raw_label: Any) -> int | None:
        if raw_label is None:
            return None
        normalized = str(raw_label).strip().lower()
        return self.LABEL_MAP.get(normalized)

    def _extract_claim(self, raw_ex: Dict[str, Any]) -> str:
        """
        Extract claim text.

        Preferred order:
        1. explicit claim field
        2. nli_fever style `premise` (claim) when `fever_gold_label` is present
        3. generic hypothesis fallback
        """
        if raw_ex.get("claim"):
            return str(raw_ex["claim"]).strip()
        if raw_ex.get("fever_gold_label") is not None and raw_ex.get("premise"):
            return str(raw_ex["premise"]).strip()
        if raw_ex.get("hypothesis"):
            return str(raw_ex["hypothesis"]).strip()
        return ""

    def _extract_evidence_texts(self, raw_ex: Dict[str, Any]) -> List[str]:
        # Hugging Face nli_fever stores the evidence text in `hypothesis`.
        if raw_ex.get("fever_gold_label") is not None and raw_ex.get("hypothesis"):
            value = raw_ex.get("hypothesis")
            texts = self._flatten_texts(value)
        else:
            texts = []
        direct_fields = (
            raw_ex.get("evidence_text"),
            raw_ex.get("premise"),
            raw_ex.get("context"),
            raw_ex.get("evidence"),
        )

        for field in direct_fields:
            texts.extend(self._flatten_texts(field))

        unique_texts = []
        seen = set()
        for text in texts:
            text = text.strip()
            if not text or text in seen:
                continue
            unique_texts.append(text)
            seen.add(text)
        return unique_texts

    def _flatten_texts(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            collected: List[str] = []
            for key in self.TEXT_KEYS:
                nested = value.get(key)
                if nested is not None:
                    collected.extend(self._flatten_texts(nested))
            return collected
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            collected = []
            for item in value:
                collected.extend(self._flatten_texts(item))
            return collected
        return []
