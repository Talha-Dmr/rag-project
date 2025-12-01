"""
PyTorch Dataset for NLI (Natural Language Inference) format data.

Handles loading, tokenization, and batching of premise-hypothesis pairs
for 3-way classification (entailment/neutral/contradiction).
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class NLIDataset(Dataset):
    """
    PyTorch Dataset for NLI training data.

    Expected data format (JSONL):
    {
        "premise": str,
        "hypothesis": str,
        "label": int,  # 0=entailment, 1=neutral, 2=contradiction
        "metadata": {...}  # optional
    }
    """

    # Label mapping
    LABEL_MAP = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }

    LABEL_NAMES = ['entailment', 'neutral', 'contradiction']

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "microsoft/deberta-v3-large-mnli",
        max_length: int = 256,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize NLI dataset.

        Args:
            data_path: Path to JSONL file with NLI examples
            tokenizer_name: Name of HuggingFace tokenizer to use
            max_length: Maximum sequence length for tokenization
            cache_dir: Optional cache directory for tokenizer
        """
        self.data_path = Path(data_path)
        self.max_length = max_length

        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir
        )

        # Load data
        logger.info(f"Loading data from: {data_path}")
        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} examples")

        # Compute label distribution
        self._log_label_distribution()

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load NLI examples from JSONL file.

        Returns:
            List of NLI examples
        """
        examples = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)

                    # Validate required fields
                    if 'premise' not in example or 'hypothesis' not in example:
                        logger.warning(
                            f"Line {line_num}: Missing premise or hypothesis, skipping"
                        )
                        continue

                    # Handle label (can be int or string)
                    if 'label' not in example:
                        logger.warning(f"Line {line_num}: Missing label, skipping")
                        continue

                    label = example['label']
                    if isinstance(label, str):
                        label = self.LABEL_MAP.get(label.lower())
                        if label is None:
                            logger.warning(
                                f"Line {line_num}: Invalid label string, skipping"
                            )
                            continue
                        example['label'] = label
                    elif not isinstance(label, int) or label not in [0, 1, 2]:
                        logger.warning(
                            f"Line {line_num}: Invalid label value {label}, skipping"
                        )
                        continue

                    examples.append(example)

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error: {e}")
                    continue

        return examples

    def _log_label_distribution(self) -> None:
        """Log the distribution of labels in the dataset."""
        label_counts = {0: 0, 1: 0, 2: 0}
        for example in self.examples:
            label_counts[example['label']] += 1

        total = len(self.examples)
        logger.info("Label distribution:")
        for label_idx, count in label_counts.items():
            label_name = self.LABEL_NAMES[label_idx]
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {label_name} ({label_idx}): {count} ({percentage:.1f}%)")

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example.

        Args:
            idx: Index of example

        Returns:
            Dictionary with tokenized inputs and label
        """
        example = self.examples[idx]

        # Tokenize premise and hypothesis pair
        encoding = self.tokenizer(
            example['premise'],
            example['hypothesis'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension added by return_tensors='pt'
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(example['label'], dtype=torch.long)
        }

        # Add token_type_ids if present (for some models)
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return item

    def get_label_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.

        Returns:
            Tensor of shape (3,) with weights for each class
        """
        label_counts = {0: 0, 1: 0, 2: 0}
        for example in self.examples:
            label_counts[example['label']] += 1

        total = len(self.examples)
        weights = []
        for label_idx in [0, 1, 2]:
            count = label_counts[label_idx]
            # Inverse frequency weighting
            weight = total / (3 * count) if count > 0 else 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def collate_nli_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to batch NLI examples.

    Args:
        batch: List of examples from NLIDataset

    Returns:
        Dictionary with batched tensors
    """
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    batched = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Add token_type_ids if present
    if 'token_type_ids' in batch[0]:
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        batched['token_type_ids'] = token_type_ids

    return batched


def create_dataloader(
    data_path: str,
    tokenizer_name: str = "microsoft/deberta-v3-large-mnli",
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_dir: Optional[str] = None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for NLI data.

    Args:
        data_path: Path to JSONL file with NLI examples
        tokenizer_name: Name of HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        cache_dir: Optional cache directory

    Returns:
        DataLoader instance
    """
    dataset = NLIDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_nli_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader
