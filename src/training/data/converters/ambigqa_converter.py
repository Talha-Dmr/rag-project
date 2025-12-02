"""
Converter for AmbigQA dataset to NLI format.

AmbigQA contains ambiguous questions with multiple valid interpretations.
Strategy: Generate entailment (valid answers), neutral (interpretation mismatch),
and contradiction (wrong/fabricated answers) examples.

Target multiplier: 3x (12K → 36K examples)
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.training.data.base_converter import BaseConverter

logger = logging.getLogger(__name__)


class AmbigQAConverter(BaseConverter):
    """Convert AmbigQA dataset to NLI format."""

    def __init__(self, dataset_path: str, multiplier: int = 3, seed: int = 42):
        """
        Initialize AmbigQA converter.

        Args:
            dataset_path: Path to AmbigQA directory (data/ambiguity_datasets/02_ambigqa)
            multiplier: Data augmentation multiplier (default: 3)
            seed: Random seed
        """
        super().__init__(dataset_path, multiplier, seed)
        random.seed(self.seed)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw AmbigQA data.

        Returns:
            List of raw AmbigQA examples
        """
        # AmbigQA has train_light.json, dev_light.json (light versions)
        data_files = ['train_light.json', 'dev_light.json']
        all_examples = []

        for file_name in data_files:
            file_path = self.dataset_path / file_name

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}, skipping")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_examples.extend(data)
            logger.info(f"Loaded {len(data)} examples from {file_name}")

        return all_examples

    def convert_to_nli(
        self,
        raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert AmbigQA examples to NLI format.

        For each example, generate:
        1. ENTAILMENT: Each valid QA pair interpretation
        2. NEUTRAL: Mismatched interpretation (answer from different interpretation)
        3. CONTRADICTION: Wrong answer or fabricated information

        Args:
            raw_examples: List of raw AmbigQA examples

        Returns:
            List of NLI examples
        """
        nli_examples = []

        for raw_ex in raw_examples:
            question_id = raw_ex.get('id', 'unknown')
            ambiguous_question = raw_ex.get('question', '')
            annotations = raw_ex.get('annotations', [])

            for ann in annotations:
                ann_type = ann.get('type', '')

                if ann_type == 'multipleQAs':
                    # Multiple interpretations - generate all three label types
                    qa_pairs = ann.get('qaPairs', [])

                    # 1. Generate ENTAILMENT examples (each QA pair)
                    for qa_pair in qa_pairs:
                        disamb_question = qa_pair.get('question', '')
                        answers = qa_pair.get('answer', [])

                        if not answers:
                            continue

                        # Convert answers to string
                        answer_text = self._format_answers(answers)

                        # Create entailment example
                        nli_examples.append({
                            'premise': ambiguous_question,
                            'hypothesis': f"{disamb_question} {answer_text}",
                            'label': self.LABEL_ENTAILMENT,
                            'metadata': {
                                'question_id': question_id,
                                'ann_type': ann_type,
                                'disambiguated_question': disamb_question
                            }
                        })

                    # 2. Generate NEUTRAL examples (interpretation mismatch)
                    if len(qa_pairs) >= 2:
                        for i, qa_pair in enumerate(qa_pairs):
                            # Use answer from a different interpretation
                            other_qa = random.choice(
                                [qp for j, qp in enumerate(qa_pairs) if j != i]
                            )

                            disamb_question = qa_pair.get('question', '')
                            wrong_answers = other_qa.get('answer', [])

                            if not wrong_answers:
                                continue

                            wrong_answer_text = self._format_answers(wrong_answers)

                            nli_examples.append({
                                'premise': ambiguous_question,
                                'hypothesis': f"{disamb_question} {wrong_answer_text}",
                                'label': self.LABEL_NEUTRAL,
                                'metadata': {
                                    'question_id': question_id,
                                    'ann_type': ann_type,
                                    'mismatch': True
                                }
                            })

                    # 3. Generate CONTRADICTION examples (fabricated/wrong info)
                    for qa_pair in qa_pairs:
                        disamb_question = qa_pair.get('question', '')

                        # Generate a contradictory hypothesis
                        contradiction_hypothesis = self._generate_contradiction(
                            ambiguous_question,
                            disamb_question,
                            qa_pair.get('answer', [])
                        )

                        nli_examples.append({
                            'premise': ambiguous_question,
                            'hypothesis': contradiction_hypothesis,
                            'label': self.LABEL_CONTRADICTION,
                            'metadata': {
                                'question_id': question_id,
                                'ann_type': ann_type,
                                'fabricated': True
                            }
                        })

                elif ann_type == 'singleAnswer':
                    # Single answer - generate entailment and contradiction
                    answers = ann.get('answer', [])

                    if not answers:
                        continue

                    answer_text = self._format_answers(answers)

                    # Entailment: correct answer
                    nli_examples.append({
                        'premise': ambiguous_question,
                        'hypothesis': f"The answer is {answer_text}",
                        'label': self.LABEL_ENTAILMENT,
                        'metadata': {
                            'question_id': question_id,
                            'ann_type': ann_type
                        }
                    })

                    # Contradiction: wrong answer
                    contradiction_hypothesis = self._generate_simple_contradiction(
                        ambiguous_question,
                        answer_text
                    )

                    nli_examples.append({
                        'premise': ambiguous_question,
                        'hypothesis': contradiction_hypothesis,
                        'label': self.LABEL_CONTRADICTION,
                        'metadata': {
                            'question_id': question_id,
                            'ann_type': ann_type,
                            'fabricated': True
                        }
                    })

        return nli_examples

    def _format_answers(self, answers: Any) -> str:
        """
        Format answers to string.

        Args:
            answers: List of answers or single answer

        Returns:
            Formatted answer string
        """
        if isinstance(answers, list):
            if len(answers) == 0:
                return ""
            elif len(answers) == 1:
                return str(answers[0])
            else:
                return f"{answers[0]} (also: {', '.join(str(a) for a in answers[1:3])})"
        return str(answers)

    def _generate_contradiction(
        self,
        question: str,
        disamb_question: str,
        correct_answers: List[str]
    ) -> str:
        """
        Generate contradictory hypothesis.

        Args:
            question: Original ambiguous question
            disamb_question: Disambiguated question
            correct_answers: Correct answers

        Returns:
            Contradictory hypothesis
        """
        # Simple strategies for generating contradictions
        strategies = [
            f"{disamb_question} This information is incorrect.",
            f"{disamb_question} The answer is unknown.",
            f"{disamb_question} There is no valid answer to this question.",
        ]

        return random.choice(strategies)

    def _generate_simple_contradiction(
        self,
        question: str,
        correct_answer: str
    ) -> str:
        """
        Generate simple contradiction for single-answer questions.

        Args:
            question: Question text
            correct_answer: Correct answer

        Returns:
            Contradictory hypothesis
        """
        strategies = [
            "The answer is unknown.",
            "This question cannot be answered.",
            "There is insufficient information to answer this.",
        ]

        return random.choice(strategies)

    def augment_examples(
        self,
        nli_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Custom augmentation for AmbigQA.

        Instead of simple duplication, we create variations by:
        - Rephrasing hypotheses
        - Adding context variations

        Args:
            nli_examples: Original NLI examples

        Returns:
            Augmented NLI examples
        """
        if self.multiplier <= 1:
            return nli_examples

        # Use base augmentation (duplication) for now
        # Can be enhanced with paraphrasing later
        return super().augment_examples(nli_examples)
