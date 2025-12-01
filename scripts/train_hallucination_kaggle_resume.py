"""
Script to resume hallucination model training on Kaggle from the latest checkpoint.

Usage:
    python scripts/train_hallucination_kaggle_resume.py \
        --data-dir /kaggle/working/nli_dataset \
        --checkpoint-dir /kaggle/working/checkpoints \
        --output-dir /kaggle/working/checkpoints \
        --config base_config \
        --epochs 5 \
        --batch-size 8 \
        --gradient-accumulation-steps 8 \
        --learning-rate 2e-5 \
        --mixed-precision fp16
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainers.hallucination_trainer import HallucinationTrainer
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint in the checkpoint directory.

    Prioritizes batch checkpoints (checkpoint-step-*) over epoch checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        logger.info(f"No checkpoint directory found at {checkpoint_dir}")
        return None

    # Look for batch checkpoints first (checkpoint-step-*)
    batch_checkpoints = list(checkpoint_path.glob("checkpoint-step-*"))
    if batch_checkpoints:
        # Sort by step number
        batch_checkpoints.sort(key=lambda p: int(p.name.split('-')[-1]))
        latest = batch_checkpoints[-1]
        logger.info(f"Found latest batch checkpoint: {latest.name}")
        return str(latest)

    # Fall back to epoch checkpoints
    epoch_checkpoints = list(checkpoint_path.glob("checkpoint-epoch-*"))
    if epoch_checkpoints:
        # Sort by epoch number
        epoch_checkpoints.sort(key=lambda p: int(p.name.split('-')[-1]))
        latest = epoch_checkpoints[-1]
        logger.info(f"Found latest epoch checkpoint: {latest.name}")
        return str(latest)

    # Check for best_model
    best_model = checkpoint_path / "best_model"
    if best_model.exists():
        logger.info("Found best_model checkpoint")
        return str(best_model)

    logger.info("No checkpoints found")
    return None


def main():
    parser = argparse.ArgumentParser(description="Resume hallucination model training")

    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing prepared NLI data")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory to search for checkpoints")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save new checkpoints")

    # Config
    parser.add_argument("--config", type=str, default="base_config",
                       help="Config name (without .yaml)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                       help="Total number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Enable TensorBoard logging")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    config['hyperparameters']['batch_size'] = args.batch_size
    config['hyperparameters']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    config['hyperparameters']['learning_rate'] = args.learning_rate
    config['hyperparameters']['mixed_precision'] = args.mixed_precision

    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)

    if latest_checkpoint:
        logger.info(f"Resuming from: {latest_checkpoint}")
    else:
        logger.info("No checkpoint found, starting from scratch")

    # Prepare data paths
    train_data = str(Path(args.data_dir) / "train.jsonl")
    val_data = str(Path(args.data_dir) / "val.jsonl")

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = HallucinationTrainer(config)

    # Prepare data
    logger.info("Loading data...")
    trainer.prepare_data(train_data, val_data)

    # Build model
    logger.info("Building model...")
    trainer.build_model()

    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        resume_from_checkpoint=latest_checkpoint
    )

    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {history['val_metrics'][-1] if history['val_metrics'] else 'N/A'}")


if __name__ == "__main__":
    main()
