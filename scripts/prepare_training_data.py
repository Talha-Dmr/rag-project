#!/usr/bin/env python3
"""
Data preparation script for hallucination detection training.

This script:
1. Converts all ambiguity datasets to NLI format
2. Merges and balances the datasets
3. Splits into train/val/test sets
4. Saves to JSONL format for training
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_loader import load_config
from src.training.data.converters.ambigqa_converter import AmbigQAConverter
from src.training.data.converters.asqa_converter import ASQAConverter
from src.training.data.converters.wic_converter import WiCConverter
from src.training.data.converters.clamber_converter import CLAMBERConverter
from src.training.data.converters.condambigqa_converter import CondAmbigQAConverter
from src.training.utils.data_utils import (
    split_dataset,
    balance_classes,
    compute_dataset_stats,
    save_jsonl
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_converters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all dataset converters from config.

    Args:
        config: Training configuration

    Returns:
        Dictionary mapping dataset names to converter instances
    """
    datasets_config = config.get('training', {}).get('datasets', {})
    converters = {}

    logger.info("Initializing dataset converters...")

    # AmbigQA
    if 'ambigqa' in datasets_config:
        cfg = datasets_config['ambigqa']
        converters['ambigqa'] = AmbigQAConverter(
            dataset_path=cfg['path'],
            multiplier=cfg.get('multiplier', 3),
            seed=42
        )

    # ASQA
    if 'asqa' in datasets_config:
        cfg = datasets_config['asqa']
        converters['asqa'] = ASQAConverter(
            dataset_path=cfg['path'],
            multiplier=cfg.get('multiplier', 4),
            seed=42
        )

    # WiC
    if 'wic' in datasets_config:
        cfg = datasets_config['wic']
        converters['wic'] = WiCConverter(
            dataset_path=cfg['path'],
            multiplier=cfg.get('multiplier', 2),
            seed=42
        )

    # CLAMBER
    if 'clamber' in datasets_config:
        cfg = datasets_config['clamber']
        converters['clamber'] = CLAMBERConverter(
            dataset_path=cfg['path'],
            multiplier=cfg.get('multiplier', 3),
            seed=42
        )

    # CondAmbigQA
    if 'condambigqa' in datasets_config:
        cfg = datasets_config['condambigqa']
        converters['condambigqa'] = CondAmbigQAConverter(
            dataset_path=cfg['path'],
            multiplier=cfg.get('multiplier', 3),
            seed=42
        )

    logger.info(f"Initialized {len(converters)} converters")
    return converters


def convert_datasets(converters: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert all datasets to NLI format.

    Args:
        converters: Dictionary of converter instances

    Returns:
        Dictionary mapping dataset names to NLI examples
    """
    converted_datasets = {}

    for name, converter in converters.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Converting {name} dataset...")
        logger.info(f"{'='*80}")

        try:
            nli_examples = converter.convert()
            converted_datasets[name] = nli_examples

            # Log statistics
            converter.log_statistics(nli_examples)

        except Exception as e:
            logger.error(f"Failed to convert {name}: {e}", exc_info=True)
            continue

    return converted_datasets


def merge_datasets(
    converted_datasets: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Merge all converted datasets with optional weighting.

    Args:
        converted_datasets: Dictionary of converted datasets
        config: Training configuration

    Returns:
        Merged list of NLI examples
    """
    logger.info("\nMerging datasets...")

    datasets_config = config.get('training', {}).get('datasets', {})
    merged = []

    for name, examples in converted_datasets.items():
        weight = datasets_config.get(name, {}).get('weight', 1.0)

        # Apply weight by sampling
        import random
        random.seed(42)

        num_samples = int(len(examples) * weight)
        if num_samples > len(examples):
            # Oversample
            sampled = random.choices(examples, k=num_samples)
        elif num_samples < len(examples):
            # Undersample
            sampled = random.sample(examples, num_samples)
        else:
            sampled = examples

        merged.extend(sampled)
        logger.info(
            f"  {name}: {len(examples)} examples × {weight} weight = "
            f"{len(sampled)} samples"
        )

    # Shuffle merged dataset
    import random
    random.seed(42)
    random.shuffle(merged)

    logger.info(f"\nMerged dataset size: {len(merged)} examples")
    return merged


def prepare_training_data(args: argparse.Namespace) -> None:
    """
    Main data preparation pipeline.

    Args:
        args: Command-line arguments
    """
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    training_config = config.get('training', {})
    data_config = training_config.get('data', {})

    # Initialize converters
    converters = load_converters(config)

    if not converters:
        logger.error("No converters initialized. Check your config.")
        sys.exit(1)

    # Convert datasets
    converted_datasets = convert_datasets(converters)

    if not converted_datasets:
        logger.error("No datasets converted successfully.")
        sys.exit(1)

    # Merge datasets
    merged_data = merge_datasets(converted_datasets, config)

    # Balance classes if requested
    if args.balance_classes or data_config.get('balance_classes', False):
        logger.info("\nBalancing classes...")
        strategy = data_config.get('balance_strategy', 'undersample')
        merged_data = balance_classes(merged_data, strategy=strategy)

    # Split into train/val/test
    logger.info("\nSplitting dataset...")
    train_split = data_config.get('train_split', 0.85)
    val_split = data_config.get('val_split', 0.10)
    test_split = data_config.get('test_split', 0.05)

    train_data, val_data, test_data = split_dataset(
        merged_data,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        shuffle=True,
        seed=42
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    logger.info(f"\nSaving datasets to: {output_dir}")

    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'
    test_path = output_dir / 'test.jsonl'

    save_jsonl(train_data, str(train_path))
    save_jsonl(val_data, str(val_path))
    save_jsonl(test_data, str(test_path))

    # Compute and save statistics
    logger.info("\nComputing dataset statistics...")

    stats = {
        'total_examples': len(merged_data),
        'train': compute_dataset_stats(train_data),
        'val': compute_dataset_stats(val_data),
        'test': compute_dataset_stats(test_data),
        'source_datasets': {
            name: len(examples)
            for name, examples in converted_datasets.items()
        }
    }

    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved statistics to: {stats_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total examples: {len(merged_data)}")
    logger.info(f"  Train: {len(train_data)} ({train_split*100:.0f}%)")
    logger.info(f"  Val:   {len(val_data)} ({val_split*100:.0f}%)")
    logger.info(f"  Test:  {len(test_data)} ({test_split*100:.0f}%)")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  {train_path}")
    logger.info(f"  {val_path}")
    logger.info(f"  {test_path}")
    logger.info(f"  {stats_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare training data for hallucination detection'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/training/nli_dataset',
        help='Output directory for prepared data'
    )

    parser.add_argument(
        '--balance-classes',
        action='store_true',
        help='Balance classes (override config)'
    )

    args = parser.parse_args()

    try:
        prepare_training_data(args)
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
