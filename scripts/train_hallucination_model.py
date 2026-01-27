#!/usr/bin/env python3
"""
Training script for hallucination detection model.

Fine-tunes DeBERTa-large on NLI data for hallucination detection.
Supports GPU acceleration, mixed precision, checkpointing, and TensorBoard logging.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_loader import load_config
from src.training.base_trainer import TrainerFactory
from src.training.trainers import hallucination_trainer  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_tensorboard(log_dir: str):
    """
    Setup TensorBoard logging.

    Args:
        log_dir: Directory for TensorBoard logs
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(log_path))
        logger.info(f"TensorBoard logging enabled: {log_path}")
        logger.info(f"Run: tensorboard --logdir={log_path}")

        return writer
    except ImportError:
        logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        return None


def train_model(args: argparse.Namespace) -> None:
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    training_config = config.get('training', {})

    # Override config with command-line arguments
    if args.epochs:
        training_config['hyperparameters']['max_epochs'] = args.epochs
    if args.batch_size:
        training_config['hyperparameters']['batch_size'] = args.batch_size
    if args.learning_rate:
        training_config['hyperparameters']['learning_rate'] = args.learning_rate
    if args.mixed_precision:
        training_config['hyperparameters']['mixed_precision'] = args.mixed_precision

    # Data paths
    data_dir = Path(args.data_dir)
    train_data = data_dir / 'train.jsonl'
    val_data = data_dir / 'val.jsonl'

    if not train_data.exists():
        logger.error(f"Training data not found: {train_data}")
        logger.error("Run prepare_training_data.py first!")
        sys.exit(1)

    if not val_data.exists():
        logger.error(f"Validation data not found: {val_data}")
        sys.exit(1)

    # Setup TensorBoard
    tensorboard_writer = None
    if args.tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(training_config.get('output', {}).get('logs_dir', './logs/training'))
        log_dir = log_dir / f'run_{timestamp}'
        tensorboard_writer = setup_tensorboard(str(log_dir))

    # Create trainer
    logger.info("Creating hallucination trainer...")
    trainer = TrainerFactory.create('hallucination', config=training_config)

    # Prepare data
    logger.info("Preparing datasets...")
    trainer.prepare_data(
        train_data_path=str(train_data),
        val_data_path=str(val_data)
    )

    # Build model
    logger.info("Building model...")
    trainer.build_model()

    # Print training info
    logger.info("\n" + "="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model: {training_config.get('model', {}).get('base_model')}")
    logger.info(f"Epochs: {training_config.get('hyperparameters', {}).get('max_epochs')}")
    logger.info(f"Batch size: {training_config.get('hyperparameters', {}).get('batch_size')}")
    logger.info(f"Gradient accumulation: {training_config.get('hyperparameters', {}).get('gradient_accumulation_steps')}")
    logger.info(f"Learning rate: {training_config.get('hyperparameters', {}).get('learning_rate')}")
    logger.info(f"Mixed precision: {training_config.get('hyperparameters', {}).get('mixed_precision')}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80 + "\n")

    # Train model
    logger.info("Starting training...")

    try:
        history = trainer.train(
            num_epochs=training_config.get('hyperparameters', {}).get('max_epochs', 5),
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from if args.resume_from else None
        )

        # Log to TensorBoard
        if tensorboard_writer:
            for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
                tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
                tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)

                # Log metrics
                val_metrics = history['val_metrics'][epoch]
                for metric_name, metric_value in val_metrics.items():
                    tensorboard_writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)

            tensorboard_writer.close()

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Best model saved to: {args.output_dir}")

        # Print final metrics
        final_metrics = history['val_metrics'][-1]
        logger.info("\nFinal Validation Metrics:")
        logger.info(f"  Accuracy: {final_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1 (macro): {final_metrics.get('f1_macro', 0):.4f}")
        logger.info(f"  F1 (weighted): {final_metrics.get('f1_weighted', 0):.4f}")
        if 'ece' in final_metrics:
            logger.info(f"  ECE: {final_metrics.get('ece', 0):.4f}")
        if 'brier' in final_metrics:
            logger.info(f"  Brier: {final_metrics.get('brier', 0):.4f}")

        logger.info("\nPer-class F1 scores:")
        for label in ['entailment', 'neutral', 'contradiction']:
            f1_key = f'f1_{label}'
            if f1_key in final_metrics:
                logger.info(f"  {label}: {final_metrics[f1_key]:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train hallucination detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing train.jsonl and val.jsonl'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save model checkpoints'
    )

    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size (overrides config)'
    )

    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        help='Gradient accumulation steps (overrides config)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--mixed-precision',
        type=str,
        choices=['fp16', 'fp32'],
        help='Mixed precision training (overrides config)'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        help='Resume training from checkpoint'
    )

    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Run training
    train_model(args)


if __name__ == '__main__':
    main()
