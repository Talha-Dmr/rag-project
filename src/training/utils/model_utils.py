"""
Model utilities for loading and initializing models.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 3,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, any]:
    """
    Load pre-trained model and tokenizer.

    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        cache_dir: Optional cache directory
        device: Device to load model on ('cuda', 'cpu', or None for auto)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir
    )

    # Move to device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")

    return model, tokenizer


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(
    model: torch.nn.Module,
    num_layers_to_freeze: int
) -> None:
    """
    Freeze bottom N layers of transformer model.

    Args:
        model: Transformer model
        num_layers_to_freeze: Number of layers to freeze from bottom
    """
    # For DeBERTa models, freeze encoder layers
    if hasattr(model, 'deberta'):
        encoder = model.deberta.encoder
        if hasattr(encoder, 'layer'):
            total_layers = len(encoder.layer)
            num_to_freeze = min(num_layers_to_freeze, total_layers)

            for i in range(num_to_freeze):
                for param in encoder.layer[i].parameters():
                    param.requires_grad = False

            logger.info(f"Froze {num_to_freeze}/{total_layers} encoder layers")
    else:
        logger.warning("Model structure not recognized for layer freezing")


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')

    Returns:
        Optimizer instance
    """
    # Separate parameters with and without weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': 0.0
        }
    ]

    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    scheduler_type: str = "linear"
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        scheduler_type: Type of scheduler ('linear', 'cosine', 'constant')

    Returns:
        Scheduler instance
    """
    from transformers import get_scheduler as hf_get_scheduler

    scheduler = hf_get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(
        f"Created {scheduler_type} scheduler "
        f"(warmup={num_warmup_steps}, total={num_training_steps})"
    )

    return scheduler


def setup_mixed_precision(enabled: bool = True):
    """
    Setup mixed precision training.

    Args:
        enabled: Whether to enable mixed precision

    Returns:
        GradScaler if enabled, None otherwise
    """
    if enabled and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Mixed precision (fp16) enabled")
        return scaler
    else:
        logger.info("Mixed precision disabled")
        return None
