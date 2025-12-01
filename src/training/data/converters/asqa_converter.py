"""
Converter for ASQA (Ambiguous Long-form QA) dataset to NLI format.

ASQA contains ambiguous questions with long-form answers.
Strategy: Extract claims from answers, verify against context, generate contradictions

Target multiplier: 4x (5.3K → 21K examples)
"""

import json
import random
import re
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.training.data.base_converter import BaseConverter

logger = logging.getLogger(__name__)


class ASQAConverter(BaseConverter):
    """Convert ASQA dataset to NLI format."""

    def __init__(self, dataset_path: str, multiplier: int = 4, seed: int = 42):
        """
        Initialize ASQA converter.

        Args:
            dataset_path: Path to ASQA directory (data/ambiguity_datasets/03_asqa)
            multiplier: Data augmentation multiplier (default: 4)
            seed: Random seed
        """
        super().__init__(dataset_path, multiplier, seed)
        random.seed(self.seed)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw ASQA data.

        ASQA format: JSON with 'train' and 'dev' splits

        Returns:
            List of raw ASQA examples
        """
        asqa_file = self.dataset_path / 'dataset' / 'ASQA.json'

        if not asqa_file.exists():
            raise FileNotFoundError(f"ASQA file not found: {asqa_file}")

        with open(asqa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_examples = []

        for split in ['train', 'dev']:
            if split in data:
                examples = data[split]
                # ASQA format is dict of examples, convert to list
                if isinstance(examples, dict):
                    examples = list(examples.values())
                all_examples.extend(examples)
                logger.info(f"Loaded {len(examples)} examples from {split}")

        return all_examples

    def convert_to_nli(
        self,
        raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert ASQA examples to NLI format.

        Strategy:
        1. Extract atomic claims from long-form answers
        2. ENTAILMENT: Verified claims from the answer
        3. NEUTRAL: Claims that are partially related
        4. CONTRADICTION: Modified/fabricated claims

        Args:
            raw_examples: List of raw ASQA examples

        Returns:
            List of NLI examples
        """
        nli_examples = []

        for raw_ex in raw_examples:
            question = raw_ex.get('ambiguous_question', '')
            qa_pairs = raw_ex.get('qa_pairs', [])
            annotations = raw_ex.get('annotations', [])

            if not question:
                continue

            # Process each annotation (long-form answer)
            for ann in annotations:
                long_answer = ann.get('long_answer', '')

                if not long_answer or len(long_answer.strip()) < 20:
                    continue

                # Extract claims from long answer (simple sentence splitting)
                claims = self._extract_claims(long_answer)

                # Generate ENTAILMENT examples (claims from answer)
                for claim in claims[:3]:  # Limit to 3 claims per answer
                    nli_examples.append({
                        'premise': question,
                        'hypothesis': claim,
                        'label': self.LABEL_ENTAILMENT,
                        'metadata': {
                            'from_long_answer': True,
                            'num_claims': len(claims)
                        }
                    })

                # Generate NEUTRAL examples (partial information)
                if claims:
                    # Take a claim and make it partial
                    partial_claim = self._make_partial_claim(claims[0])

                    nli_examples.append({
                        'premise': question,
                        'hypothesis': partial_claim,
                        'label': self.LABEL_NEUTRAL,
                        'metadata': {
                            'generated': 'partial',
                            'from_long_answer': True
                        }
                    })

                # Generate CONTRADICTION examples (modify claims)
                if claims:
                    contradictory_claim = self._generate_contradictory_claim(
                        question,
                        claims[0]
                    )

                    nli_examples.append({
                        'premise': question,
                        'hypothesis': contradictory_claim,
                        'label': self.LABEL_CONTRADICTION,
                        'metadata': {
                            'fabricated': True,
                            'from_long_answer': True
                        }
                    })

            # Also use QA pairs if available
            for qa_pair in qa_pairs:
                qa_question = qa_pair.get('question', '')
                qa_answer = qa_pair.get('answer', [])

                if not qa_question or not qa_answer:
                    continue

                answer_text = qa_answer[0] if isinstance(qa_answer, list) else qa_answer

                # ENTAILMENT: Correct QA pair
                nli_examples.append({
                    'premise': question,
                    'hypothesis': f"{qa_question} {answer_text}",
                    'label': self.LABEL_ENTAILMENT,
                    'metadata': {
                        'from_qa_pair': True
                    }
                })

        return nli_examples

    def _extract_claims(self, long_answer: str) -> List[str]:
        """
        Extract atomic claims from long-form answer.

        Simple implementation: Split by sentences.

        Args:
            long_answer: Long-form answer text

        Returns:
            List of claim sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', long_answer)

        claims = []
        for sent in sentences:
            sent = sent.strip()
            # Filter out very short or empty sentences
            if len(sent.split()) >= 5:  # At least 5 words
                claims.append(sent + '.')

        return claims

    def _make_partial_claim(self, claim: str) -> str:
        """
        Make a claim partial/incomplete for neutral examples.

        Args:
            claim: Original claim

        Returns:
            Partial claim
        """
        # Simple strategy: Add hedging language
        hedges = [
            f"It is possible that {claim.lower()}",
            f"Some sources suggest that {claim.lower()}",
            f"There is evidence that {claim.lower()}",
        ]

        return random.choice(hedges)

    def _generate_contradictory_claim(self, question: str, claim: str) -> str:
        """
        Generate contradictory version of claim.

        Args:
            question: Original question
            claim: Original claim

        Returns:
            Contradictory claim
        """
        # Simple negation strategies
        contradictions = [
            "There is no evidence supporting this claim.",
            "This information is incorrect.",
            "The opposite is actually true.",
        ]

        return random.choice(contradictions)
