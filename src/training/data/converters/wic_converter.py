"""
Converter for WiC (Word-in-Context) dataset to NLI format.

WiC contains word pairs in different sentences for word sense disambiguation.
Strategy: Same sense (T) → Entailment, Different sense (F) → Contradiction

Target multiplier: 2x (6K → 12K examples)
"""

import random
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.training.data.base_converter import BaseConverter

logger = logging.getLogger(__name__)


class WiCConverter(BaseConverter):
    """Convert WiC dataset to NLI format."""

    def __init__(self, dataset_path: str, multiplier: int = 2, seed: int = 42):
        """
        Initialize WiC converter.

        Args:
            dataset_path: Path to WiC directory (data/ambiguity_datasets/01_wic)
            multiplier: Data augmentation multiplier (default: 2)
            seed: Random seed
        """
        super().__init__(dataset_path, multiplier, seed)
        random.seed(self.seed)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw WiC data.

        WiC format:
        - data.txt: target_word \t pos \t indices \t sentence1 \t sentence2
        - gold.txt: T (same sense) or F (different sense)

        Returns:
            List of raw WiC examples
        """
        all_examples = []
        splits = ['train', 'dev']

        for split in splits:
            split_dir = self.dataset_path / split
            data_file = split_dir / f"{split}.data.txt"
            gold_file = split_dir / f"{split}.gold.txt"

            if not data_file.exists() or not gold_file.exists():
                logger.warning(f"Files not found for split {split}, skipping")
                continue

            # Read data and labels
            with open(data_file, 'r', encoding='utf-8') as f:
                data_lines = f.readlines()

            with open(gold_file, 'r', encoding='utf-8') as f:
                gold_lines = f.readlines()

            # Parse examples
            for data_line, gold_line in zip(data_lines, gold_lines):
                parts = data_line.strip().split('\t')

                if len(parts) >= 5:
                    example = {
                        'target_word': parts[0],
                        'pos_tag': parts[1],
                        'indices': parts[2],
                        'sentence1': parts[3],
                        'sentence2': parts[4],
                        'label': gold_line.strip()  # 'T' or 'F'
                    }
                    all_examples.append(example)

            logger.info(f"Loaded {len(data_lines)} examples from {split}")

        return all_examples

    def convert_to_nli(
        self,
        raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert WiC examples to NLI format.

        Strategy:
        1. ENTAILMENT: Same sense (label='T') - word means same in both sentences
        2. CONTRADICTION: Different sense (label='F') - word has different meanings
        3. NEUTRAL: Generate ambiguous cases or partial matches

        Args:
            raw_examples: List of raw WiC examples

        Returns:
            List of NLI examples
        """
        nli_examples = []

        for raw_ex in raw_examples:
            target_word = raw_ex['target_word']
            sentence1 = raw_ex['sentence1']
            sentence2 = raw_ex['sentence2']
            label = raw_ex['label']  # 'T' or 'F'

            # Premise: Context about the word usage
            premise = (
                f"The word '{target_word}' is used in these two contexts: "
                f"(1) {sentence1} (2) {sentence2}"
            )

            # Hypothesis: Statement about word sense
            if label == 'T':
                # Same sense → ENTAILMENT
                hypothesis = (
                    f"The word '{target_word}' has the same meaning "
                    f"in both contexts."
                )
                nli_label = self.LABEL_ENTAILMENT

            else:  # label == 'F'
                # Different sense → First create as CONTRADICTION
                hypothesis = (
                    f"The word '{target_word}' has the same meaning "
                    f"in both contexts."
                )
                nli_label = self.LABEL_CONTRADICTION

            nli_examples.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'label': nli_label,
                'metadata': {
                    'target_word': target_word,
                    'original_label': label,
                    'pos_tag': raw_ex['pos_tag']
                }
            })

            # Generate NEUTRAL examples (uncertainty about word sense)
            if random.random() < 0.5:  # 50% chance to generate neutral
                neutral_hypothesis = (
                    f"The word '{target_word}' might have related "
                    f"but not identical meanings in both contexts."
                )

                nli_examples.append({
                    'premise': premise,
                    'hypothesis': neutral_hypothesis,
                    'label': self.LABEL_NEUTRAL,
                    'metadata': {
                        'target_word': target_word,
                        'original_label': label,
                        'pos_tag': raw_ex['pos_tag'],
                        'generated': 'neutral'
                    }
                })

        return nli_examples
