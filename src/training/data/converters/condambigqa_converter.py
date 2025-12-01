"""
Converter for CondAmbigQA-2K dataset to NLI format.

CondAmbigQA contains questions with conditional properties/contexts.
Strategy: Correct property with context → Entailment,
         Wrong value → Contradiction, Property confusion → Neutral

Target multiplier: 3x (2K → 6K examples)
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.training.data.base_converter import BaseConverter

logger = logging.getLogger(__name__)


class CondAmbigQAConverter(BaseConverter):
    """Convert CondAmbigQA dataset to NLI format."""

    def __init__(self, dataset_path: str, multiplier: int = 3, seed: int = 42):
        """
        Initialize CondAmbigQA converter.

        Args:
            dataset_path: Path to CondAmbigQA directory (data/ambiguity_datasets/05_condambigqa)
            multiplier: Data augmentation multiplier (default: 3)
            seed: Random seed
        """
        super().__init__(dataset_path, multiplier, seed)
        random.seed(self.seed)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw CondAmbigQA data.

        CondAmbigQA format: JSON files for each split

        Returns:
            List of raw CondAmbigQA examples
        """
        all_examples = []
        data_files = ['train.json', 'validation.json']

        for file_name in data_files:
            file_path = self.dataset_path / file_name

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}, skipping")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # CondAmbigQA is a list of examples
            if isinstance(data, list):
                all_examples.extend(data)
            elif isinstance(data, dict):
                # If it's a dict, might have 'data' or similar key
                all_examples.extend(data.get('data', data.values()))

            logger.info(f"Loaded {len(data)} examples from {file_name}")

        return all_examples

    def convert_to_nli(
        self,
        raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert CondAmbigQA examples to NLI format.

        Strategy:
        1. ENTAILMENT: Correct property/answer with proper context
        2. CONTRADICTION: Wrong value for property
        3. NEUTRAL: Property confusion or context mismatch

        Args:
            raw_examples: List of raw CondAmbigQA examples

        Returns:
            List of NLI examples
        """
        nli_examples = []

        for raw_ex in raw_examples:
            # CondAmbigQA structure varies, adapt to what we find
            question = raw_ex.get('question', raw_ex.get('Question', ''))
            properties = raw_ex.get('properties', raw_ex.get('Properties', {}))
            contexts = raw_ex.get('contexts', raw_ex.get('Contexts', []))

            if not question:
                continue

            # Process properties
            if isinstance(properties, dict):
                for prop_name, prop_value in properties.items():
                    # 1. ENTAILMENT: Correct property
                    hypothesis_correct = (
                        f"For the question '{question}', "
                        f"the {prop_name} is {prop_value}."
                    )

                    nli_examples.append({
                        'premise': question,
                        'hypothesis': hypothesis_correct,
                        'label': self.LABEL_ENTAILMENT,
                        'metadata': {
                            'property': prop_name,
                            'value': str(prop_value),
                            'type': 'correct'
                        }
                    })

                    # 2. CONTRADICTION: Wrong value
                    hypothesis_wrong = (
                        f"For the question '{question}', "
                        f"the {prop_name} is unknown."
                    )

                    nli_examples.append({
                        'premise': question,
                        'hypothesis': hypothesis_wrong,
                        'label': self.LABEL_CONTRADICTION,
                        'metadata': {
                            'property': prop_name,
                            'value': str(prop_value),
                            'type': 'wrong_value'
                        }
                    })

            # 3. NEUTRAL: Context confusion
            if contexts and len(contexts) > 0:
                context_text = contexts[0] if isinstance(contexts, list) else str(contexts)

                hypothesis_context = (
                    f"Additional context may be needed to fully answer: {question}"
                )

                nli_examples.append({
                    'premise': question,
                    'hypothesis': hypothesis_context,
                    'label': self.LABEL_NEUTRAL,
                    'metadata': {
                        'has_context': True,
                        'type': 'context_dependent'
                    }
                })

        return nli_examples
