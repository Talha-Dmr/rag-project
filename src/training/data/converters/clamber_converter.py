"""
Converter for CLAMBER (Clarification Ambiguity Benchmark) dataset to NLI format.

CLAMBER contains ambiguous queries requiring clarification.
Strategy: Ambiguous without clarification → Neutral,
         With clarification → Entailment, Wrong assumption → Contradiction

Target multiplier: 3x (3.2K → 9.6K examples)
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.training.data.base_converter import BaseConverter

logger = logging.getLogger(__name__)


class CLAMBERConverter(BaseConverter):
    """Convert CLAMBER dataset to NLI format."""

    def __init__(self, dataset_path: str, multiplier: int = 3, seed: int = 42):
        """
        Initialize CLAMBER converter.

        Args:
            dataset_path: Path to CLAMBER directory (data/ambiguity_datasets/04_clamber)
            multiplier: Data augmentation multiplier (default: 3)
            seed: Random seed
        """
        super().__init__(dataset_path, multiplier, seed)
        random.seed(self.seed)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw CLAMBER data.

        CLAMBER format: JSONL with double-encoding

        Returns:
            List of raw CLAMBER examples
        """
        all_examples = []
        # CLAMBER has single file: clamber_benchmark.jsonl
        data_files = ['clamber_benchmark.jsonl']

        for file_name in data_files:
            file_path = self.dataset_path / file_name

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}, skipping")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Double-encoded JSON
                    try:
                        json_str = json.loads(line)
                        json_obj = json.loads(json_str)
                        all_examples.append(json_obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue

            logger.info(f"Loaded examples from {file_name}")

        return all_examples

    def convert_to_nli(
        self,
        raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert CLAMBER examples to NLI format.

        Strategy:
        1. NEUTRAL: Ambiguous query without clarification (insufficient context)
        2. ENTAILMENT: Query with correct clarification
        3. CONTRADICTION: Query with wrong assumption/clarification

        Args:
            raw_examples: List of raw CLAMBER examples

        Returns:
            List of NLI examples
        """
        nli_examples = []

        for raw_ex in raw_examples:
            initial_request = raw_ex.get('initial_request', '')
            clarification_need = raw_ex.get('clarification_need', '')
            facet = raw_ex.get('facet', '')

            if not initial_request:
                continue

            # 1. NEUTRAL: Ambiguous without clarification
            hypothesis_ambiguous = (
                f"The request '{initial_request}' can be answered directly "
                f"without additional information."
            )

            nli_examples.append({
                'premise': initial_request,
                'hypothesis': hypothesis_ambiguous,
                'label': self.LABEL_NEUTRAL,
                'metadata': {
                    'clarification_need': clarification_need,
                    'facet': facet,
                    'type': 'ambiguous'
                }
            })

            # 2. ENTAILMENT: With clarification (if available)
            if clarification_need:
                hypothesis_clarified = (
                    f"To answer '{initial_request}', we need to know: "
                    f"{clarification_need}"
                )

                nli_examples.append({
                    'premise': initial_request,
                    'hypothesis': hypothesis_clarified,
                    'label': self.LABEL_ENTAILMENT,
                    'metadata': {
                        'clarification_need': clarification_need,
                        'facet': facet,
                        'type': 'clarified'
                    }
                })

            # 3. CONTRADICTION: Wrong assumption
            hypothesis_wrong = (
                f"The request '{initial_request}' is completely clear "
                f"and unambiguous."
            )

            nli_examples.append({
                'premise': initial_request,
                'hypothesis': hypothesis_wrong,
                'label': self.LABEL_CONTRADICTION,
                'metadata': {
                    'clarification_need': clarification_need,
                    'facet': facet,
                    'type': 'wrong_assumption'
                }
            })

        return nli_examples
