#!/usr/bin/env python3
"""
Export script for hallucination detection model.

Exports trained model to production-ready format for inference.
Saves model in HuggingFace format for easy loading.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_loader import load_config
from src.training.base_trainer import TrainerFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_model(args: argparse.Namespace) -> None:
    """
    Export trained model for production.

    Args:
        args: Command-line arguments
    """
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    training_config = config.get('training', {})

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("EXPORTING MODEL FOR PRODUCTION")
    logger.info("="*80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_dir}")

    # Create trainer
    logger.info("\nCreating trainer...")
    trainer = TrainerFactory.create('hallucination', config=training_config)

    # Build model
    logger.info("Building model...")
    trainer.build_model()

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    state = trainer.load_checkpoint(str(checkpoint_path))

    logger.info(f"Loaded checkpoint from epoch {state.get('epoch', 'unknown')}")

    # Set model to eval mode
    trainer.model.eval()

    # Save model in HuggingFace format
    logger.info("\nSaving model in HuggingFace format...")

    # Save model weights
    model_save_path = output_dir / "pytorch_model.bin"
    torch.save(trainer.model.state_dict(), model_save_path)
    logger.info(f"Saved model weights: {model_save_path}")

    # Save full model (for easier loading)
    full_model_path = output_dir / "model"
    trainer.model.save_pretrained(str(full_model_path))
    logger.info(f"Saved HuggingFace model: {full_model_path}")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer"
    trainer.tokenizer.save_pretrained(str(tokenizer_path))
    logger.info(f"Saved tokenizer: {tokenizer_path}")

    # Save config
    model_config = {
        'model_type': 'deberta-v2',
        'task': 'text-classification',
        'num_labels': 3,
        'id2label': {
            0: 'entailment',
            1: 'neutral',
            2: 'contradiction'
        },
        'label2id': {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        },
        'base_model': training_config.get('model', {}).get('base_model'),
        'max_seq_length': training_config.get('data', {}).get('max_seq_length', 256),
        'training_metrics': state.get('metrics', {})
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved model config: {config_path}")

    # Optimize for inference if requested
    if args.optimize_inference:
        logger.info("\nOptimizing for inference...")

        # Convert to half precision for GPU inference (optional)
        if args.half_precision and torch.cuda.is_available():
            trainer.model.half()
            logger.info("Converted to half precision (fp16)")

        # Save optimized model
        optimized_path = output_dir / "model_optimized.pt"
        torch.save(trainer.model.state_dict(), optimized_path)
        logger.info(f"Saved optimized model: {optimized_path}")

    # Create inference example script
    example_script = '''#!/usr/bin/env python3
"""
Example inference script for hallucination detection.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "."  # Current directory
model = AutoModelForSequenceClassification.from_pretrained(model_path + "/model")
tokenizer = AutoTokenizer.from_pretrained(model_path + "/tokenizer")

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Example usage
premise = "When did the Simpsons first air?"
hypothesis = "The Simpsons first aired in 1989."

# Tokenize
inputs = tokenizer(
    premise,
    hypothesis,
    max_length=256,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

# Map to label
id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
label = id2label[predicted_class]
confidence = predictions[0][predicted_class].item()

print(f"Premise: {premise}")
print(f"Hypothesis: {hypothesis}")
print(f"Prediction: {label} (confidence: {confidence:.4f})")
print(f"Scores: {predictions[0].tolist()}")
'''

    example_path = output_dir / "example_inference.py"
    with open(example_path, 'w') as f:
        f.write(example_script)
    logger.info(f"Created example script: {example_path}")

    # Create README
    readme_content = f'''# Hallucination Detection Model

Fine-tuned DeBERTa-large model for hallucination detection in RAG systems.

## Model Details
- Base Model: {training_config.get('model', {}).get('base_model')}
- Task: 3-way NLI classification
- Labels: entailment (0), neutral (1), contradiction (2)

## Files
- `model/` - HuggingFace model weights
- `tokenizer/` - Tokenizer configuration
- `config.json` - Model configuration
- `example_inference.py` - Example usage script

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

# Your inference code here
```

## Performance Metrics
{json.dumps(state.get('metrics', {}), indent=2)}

## Training Info
- Checkpoint Epoch: {state.get('epoch', 'unknown')}
- Exported: {output_dir}
'''

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    logger.info(f"Created README: {readme_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPORT COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model exported to: {output_dir}")
    logger.info("\nFiles created:")
    logger.info(f"  - model/ (HuggingFace format)")
    logger.info(f"  - tokenizer/")
    logger.info(f"  - config.json")
    logger.info(f"  - example_inference.py")
    logger.info(f"  - README.md")
    logger.info(f"\nRun example: python {example_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export hallucination detection model for production',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save exported model'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--optimize-inference',
        action='store_true',
        help='Apply inference optimizations'
    )

    parser.add_argument(
        '--half-precision',
        action='store_true',
        help='Convert to half precision (fp16) for GPU inference'
    )

    args = parser.parse_args()

    try:
        export_model(args)
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
