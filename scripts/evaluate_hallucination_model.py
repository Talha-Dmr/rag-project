#!/usr/bin/env python3
"""
Evaluation script for trained hallucination detection model.

Evaluates model on test set and generates comprehensive evaluation report
including metrics, confusion matrix, and per-class performance.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_loader import load_config
from src.training.base_trainer import TrainerFactory
from src.training.metrics.nli_metrics import NLIMetrics, print_confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm, label_names, output_path):
    """
    Plot and save confusion matrix visualization.

    Args:
        cm: Confusion matrix array
        label_names: List of label names
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Hallucination Detection', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to: {output_path}")
    plt.close()


def plot_per_class_metrics(metrics, output_path):
    """
    Plot per-class metrics (F1, Precision, Recall).

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save figure
    """
    labels = ['entailment', 'neutral', 'contradiction']

    f1_scores = [metrics.get(f'f1_{label}', 0) for label in labels]
    precision_scores = [metrics.get(f'precision_{label}', 0) for label in labels]
    recall_scores = [metrics.get(f'recall_{label}', 0) for label in labels]

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar([i - width for i in x], f1_scores, width, label='F1', color='#2E86AB')
    ax.bar(x, precision_scores, width, label='Precision', color='#A23B72')
    ax.bar([i + width for i in x], recall_scores, width, label='Recall', color='#F18F01')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved per-class metrics to: {output_path}")
    plt.close()


def evaluate_model(args: argparse.Namespace) -> None:
    """
    Main evaluation function.

    Args:
        args: Command-line arguments
    """
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    training_config = config.get('training', {})

    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    test_data = data_dir / 'test.jsonl'

    if not test_data.exists():
        logger.error(f"Test data not found: {test_data}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = TrainerFactory.create('hallucination', config=training_config)

    # Build model
    logger.info("Building model...")
    trainer.build_model()

    # Load checkpoint
    logger.info(f"Loading model from: {model_path}")
    trainer.load_checkpoint(str(model_path))

    # Evaluate
    logger.info(f"Evaluating on test set: {test_data}")
    logger.info("="*80)

    metrics = trainer.evaluate(str(test_data))

    # Get detailed metrics
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)

    logger.info("\nOverall Metrics:")
    logger.info(f"  Accuracy:         {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro):       {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (weighted):    {metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro):    {metrics['recall_macro']:.4f}")

    logger.info("\nPer-Class Metrics:")
    for label in ['entailment', 'neutral', 'contradiction']:
        logger.info(f"\n  {label.capitalize()}:")
        logger.info(f"    F1:        {metrics[f'f1_{label}']:.4f}")
        logger.info(f"    Precision: {metrics[f'precision_{label}']:.4f}")
        logger.info(f"    Recall:    {metrics[f'recall_{label}']:.4f}")

    # Save metrics to JSON
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nSaved metrics to: {metrics_path}")

    # Generate confusion matrix (need to re-evaluate to get predictions)
    logger.info("\nGenerating confusion matrix...")

    from src.training.data.nli_dataset import create_dataloader
    import torch

    test_loader = create_dataloader(
        data_path=str(test_data),
        tokenizer_name=training_config.get('model', {}).get('base_model'),
        batch_size=32,
        max_length=training_config.get('data', {}).get('max_seq_length', 256),
        shuffle=False,
        cache_dir=training_config.get('model', {}).get('cache_dir')
    )

    metrics_tracker = NLIMetrics()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            metrics_tracker.update(
                predictions=predictions.cpu().tolist(),
                labels=labels.cpu().tolist()
            )

    # Get confusion matrix
    cm = metrics_tracker.get_confusion_matrix()

    # Print confusion matrix
    logger.info("\nConfusion Matrix:")
    print_confusion_matrix(cm)

    # Save confusion matrix visualization
    cm_plot_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, ['Entailment', 'Neutral', 'Contradiction'], cm_plot_path)

    # Save per-class metrics plot
    metrics_plot_path = output_dir / 'per_class_metrics.png'
    plot_per_class_metrics(metrics, metrics_plot_path)

    # Save classification report
    report = metrics_tracker.get_classification_report()
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved classification report to: {report_path}")

    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - {metrics_path.name}")
    logger.info(f"  - {cm_plot_path.name}")
    logger.info(f"  - {metrics_plot_path.name}")
    logger.info(f"  - {report_path.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate hallucination detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint directory'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing test.jsonl'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    try:
        evaluate_model(args)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
