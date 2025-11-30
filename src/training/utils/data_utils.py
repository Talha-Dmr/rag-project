"""
Data utilities for dataset conversion and manipulation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries
        file_path: Path to save JSONL file
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(data)} examples to {file_path}")


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train/val/test sets.

    Args:
        data: List of examples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle data before splitting
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    logger.info(f"Split dataset: train={len(train_data)}, "
                f"val={len(val_data)}, test={len(test_data)}")

    return train_data, val_data, test_data


def balance_classes(
    data: List[Dict[str, Any]],
    label_key: str = 'label',
    strategy: str = 'undersample',
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Balance dataset classes.

    Args:
        data: List of examples
        label_key: Key for label field
        strategy: 'undersample' or 'oversample'
        seed: Random seed

    Returns:
        Balanced dataset
    """
    random.seed(seed)

    # Count labels
    label_counts = Counter(ex[label_key] for ex in data)
    logger.info(f"Original label distribution: {dict(label_counts)}")

    if strategy == 'undersample':
        # Undersample to minority class size
        min_count = min(label_counts.values())

        # Group by label
        label_groups = {}
        for ex in data:
            label = ex[label_key]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(ex)

        # Sample from each group
        balanced_data = []
        for label, examples in label_groups.items():
            sampled = random.sample(examples, min(len(examples), min_count))
            balanced_data.extend(sampled)

        random.shuffle(balanced_data)

    elif strategy == 'oversample':
        # Oversample to majority class size
        max_count = max(label_counts.values())

        # Group by label
        label_groups = {}
        for ex in data:
            label = ex[label_key]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(ex)

        # Oversample each group
        balanced_data = []
        for label, examples in label_groups.items():
            # Repeat examples to reach max_count
            oversampled = examples * (max_count // len(examples))
            oversampled += random.sample(examples, max_count % len(examples))
            balanced_data.extend(oversampled)

        random.shuffle(balanced_data)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Count labels after balancing
    final_counts = Counter(ex[label_key] for ex in balanced_data)
    logger.info(f"Balanced label distribution: {dict(final_counts)}")

    return balanced_data


def compute_dataset_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics for dataset.

    Args:
        data: List of NLI examples

    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_examples': len(data),
        'label_distribution': {},
        'avg_premise_length': 0,
        'avg_hypothesis_length': 0
    }

    if not data:
        return stats

    # Label distribution
    label_counts = Counter(ex.get('label') for ex in data)
    stats['label_distribution'] = dict(label_counts)

    # Text length statistics
    premise_lengths = []
    hypothesis_lengths = []

    for ex in data:
        if 'premise' in ex:
            premise_lengths.append(len(ex['premise'].split()))
        if 'hypothesis' in ex:
            hypothesis_lengths.append(len(ex['hypothesis'].split()))

    if premise_lengths:
        stats['avg_premise_length'] = sum(premise_lengths) / len(premise_lengths)
        stats['max_premise_length'] = max(premise_lengths)

    if hypothesis_lengths:
        stats['avg_hypothesis_length'] = sum(hypothesis_lengths) / len(hypothesis_lengths)
        stats['max_hypothesis_length'] = max(hypothesis_lengths)

    return stats


def merge_datasets(
    datasets: List[List[Dict[str, Any]]],
    weights: List[float],
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Merge multiple datasets with optional weighting.

    Args:
        datasets: List of datasets to merge
        weights: Weight for each dataset
        shuffle: Whether to shuffle merged data
        seed: Random seed

    Returns:
        Merged dataset
    """
    assert len(datasets) == len(weights), \
        "Number of datasets must match number of weights"

    random.seed(seed)

    merged = []
    for dataset, weight in zip(datasets, weights):
        # Apply weight by sampling
        num_samples = int(len(dataset) * weight)
        if num_samples > len(dataset):
            # Oversample
            sampled = random.choices(dataset, k=num_samples)
        else:
            # Undersample or keep as is
            sampled = random.sample(dataset, num_samples) if num_samples < len(dataset) else dataset
        merged.extend(sampled)

    if shuffle:
        random.shuffle(merged)

    logger.info(f"Merged {len(datasets)} datasets into {len(merged)} examples")

    return merged


def augment_text(text: str, augmentation_type: str = 'synonym') -> str:
    """
    Apply text augmentation (placeholder for future implementation).

    Args:
        text: Input text
        augmentation_type: Type of augmentation

    Returns:
        Augmented text
    """
    # TODO: Implement text augmentation techniques
    # - Synonym replacement
    # - Back-translation
    # - Paraphrasing
    return text
