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

                    # Try normal JSON first, then double-encoded
                    try:
                        json_obj = json.loads(line)
                        # If it's a string, parse again (double-encoded)
                        if isinstance(json_obj, str):
                            json_obj = json.loads(json_obj)
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
            question = raw_ex.get('question', '')
            clarifying_question = raw_ex.get('clarifying_question', '')
            require_clarification = raw_ex.get('require_clarification', False)

            if not question:
                continue

            # 1. NEUTRAL: Ambiguous without clarification
            hypothesis_ambiguous = (
                f"The request '{question}' can be answered directly "
                f"without additional information."
            )

            nli_examples.append({
                'premise': question,
                'hypothesis': hypothesis_ambiguous,
                'label': self.LABEL_NEUTRAL,
                'metadata': {
                    'clarifying_question': clarifying_question,
                    'require_clarification': require_clarification,
                    'type': 'ambiguous'
                }
            })

            # 2. ENTAILMENT: With clarification (if available)
            if clarifying_question:
                hypothesis_clarified = (
                    f"To answer '{question}', we need to know: "
                    f"{clarifying_question}"
                )

                nli_examples.append({
                    'premise': question,
                    'hypothesis': hypothesis_clarified,
                    'label': self.LABEL_ENTAILMENT,
                    'metadata': {
                        'clarifying_question': clarifying_question,
                        'require_clarification': require_clarification,
                        'type': 'clarified'
                    }
                })

            # 3. CONTRADICTION: Wrong assumption
            hypothesis_wrong = (
                f"The request '{question}' is completely clear "
                f"and unambiguous."
            )

            nli_examples.append({
                'premise': question,
                'hypothesis': hypothesis_wrong,
                'label': self.LABEL_CONTRADICTION,
                'metadata': {
                    'clarifying_question': clarifying_question,
                    'require_clarification': require_clarification,
                    'type': 'wrong_assumption'
                }
            })

        return nli_examples
