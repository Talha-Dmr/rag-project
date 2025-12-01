"""
Base converter interface for converting ambiguity datasets to NLI format.

All dataset converters must inherit from this base class and implement
the conversion logic for their specific dataset format.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseConverter(ABC):
    """
    Abstract base class for dataset converters.

    Converts ambiguity detection datasets to NLI format:
    {
        "premise": str,
        "hypothesis": str,
        "label": int,  # 0=entailment, 1=neutral, 2=contradiction
        "metadata": {...}
    }
    """

    # Label constants
    LABEL_ENTAILMENT = 0
    LABEL_NEUTRAL = 1
    LABEL_CONTRADICTION = 2

    LABEL_NAMES = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    def __init__(
        self,
        dataset_path: str,
        multiplier: int = 1,
        seed: int = 42
    ):
        """
        Initialize converter.

        Args:
            dataset_path: Path to dataset directory or file
            multiplier: Data augmentation multiplier (1 = no augmentation)
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.multiplier = multiplier
        self.seed = seed
        self.dataset_name = self.__class__.__name__.replace('Converter', '')

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        logger.info(
            f"Initialized {self.dataset_name} converter: "
            f"path={dataset_path}, multiplier={multiplier}"
        )

    @abstractmethod
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw dataset from source.

        Returns:
            List of raw examples in original format

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def convert_to_nli(
        self,
        raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert raw examples to NLI format.

        Args:
            raw_examples: List of raw examples

        Returns:
            List of NLI examples with premise, hypothesis, label

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    def augment_examples(
        self,
        nli_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply data augmentation to reach target multiplier.

        Default implementation duplicates examples. Subclasses can override
        for dataset-specific augmentation strategies.

        Args:
            nli_examples: List of NLI examples

        Returns:
            Augmented list of NLI examples
        """
        if self.multiplier <= 1:
            return nli_examples

        target_size = len(nli_examples) * self.multiplier
        current_size = len(nli_examples)

        augmented = nli_examples.copy()

        import random
        random.seed(self.seed)

        # Simple duplication with shuffling
        while len(augmented) < target_size:
            remaining = target_size - len(augmented)
            if remaining >= current_size:
                augmented.extend(nli_examples)
            else:
                augmented.extend(random.sample(nli_examples, remaining))

        random.shuffle(augmented)

        logger.info(
            f"Augmented {current_size} → {len(augmented)} examples "
            f"(multiplier={self.multiplier})"
        )

        return augmented

    def validate_nli_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate that an NLI example has required fields.

        Args:
            example: NLI example to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['premise', 'hypothesis', 'label']

        for field in required_fields:
            if field not in example:
                logger.warning(f"Missing required field: {field}")
                return False

        # Validate label
        if example['label'] not in [0, 1, 2]:
            logger.warning(f"Invalid label: {example['label']}")
            return False

        # Validate text fields
        if not isinstance(example['premise'], str) or not example['premise'].strip():
            logger.warning("Invalid premise")
            return False

        if not isinstance(example['hypothesis'], str) or not example['hypothesis'].strip():
            logger.warning("Invalid hypothesis")
            return False

        return True

    def convert(self) -> List[Dict[str, Any]]:
        """
        Execute full conversion pipeline.

        Returns:
            List of NLI examples ready for training
        """
        logger.info(f"Starting conversion for {self.dataset_name}")

        # Load raw data
        raw_examples = self.load_raw_data()
        logger.info(f"Loaded {len(raw_examples)} raw examples")

        # Convert to NLI
        nli_examples = self.convert_to_nli(raw_examples)
        logger.info(f"Converted to {len(nli_examples)} NLI examples")

        # Validate examples
        valid_examples = [ex for ex in nli_examples if self.validate_nli_example(ex)]
        logger.info(
            f"Validated {len(valid_examples)}/{len(nli_examples)} examples"
        )

        # Augment if needed
        if self.multiplier > 1:
            valid_examples = self.augment_examples(valid_examples)

        # Add source metadata
        for ex in valid_examples:
            if 'metadata' not in ex:
                ex['metadata'] = {}
            ex['metadata']['source_dataset'] = self.dataset_name

        logger.info(
            f"Conversion complete: {len(valid_examples)} final NLI examples"
        )

        return valid_examples

    def get_label_distribution(
        self,
        nli_examples: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Compute label distribution in NLI examples.

        Args:
            nli_examples: List of NLI examples

        Returns:
            Dictionary mapping label names to counts
        """
        from collections import Counter

        label_counts = Counter(ex['label'] for ex in nli_examples)

        distribution = {
            self.LABEL_NAMES[label]: count
            for label, count in label_counts.items()
        }

        return distribution

    def log_statistics(self, nli_examples: List[Dict[str, Any]]) -> None:
        """
        Log statistics about converted examples.

        Args:
            nli_examples: List of NLI examples
        """
        if not nli_examples:
            logger.warning("No examples to log statistics for")
            return

        distribution = self.get_label_distribution(nli_examples)

        logger.info(f"\n{self.dataset_name} Statistics:")
        logger.info(f"  Total examples: {len(nli_examples)}")
        logger.info(f"  Label distribution:")
        for label_name, count in distribution.items():
            percentage = (count / len(nli_examples)) * 100
            logger.info(f"    {label_name}: {count} ({percentage:.1f}%)")

        # Text length statistics
        premise_lengths = [len(ex['premise'].split()) for ex in nli_examples]
        hypothesis_lengths = [len(ex['hypothesis'].split()) for ex in nli_examples]

        logger.info(f"  Average premise length: {sum(premise_lengths) / len(premise_lengths):.1f} words")
        logger.info(f"  Average hypothesis length: {sum(hypothesis_lengths) / len(hypothesis_lengths):.1f} words")
