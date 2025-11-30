"""
Base trainer interface and factory pattern for model training.

This module defines the abstract trainer interface and factory for creating
trainer instances, enabling modular and extensible training pipelines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for model trainers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer with configuration.

        Args:
            config: Trainer configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.metrics = {}

    @abstractmethod
    def prepare_data(self, train_data_path: str, val_data_path: str) -> None:
        """
        Prepare training and validation datasets.

        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
        """
        pass

    @abstractmethod
    def build_model(self) -> None:
        """
        Build and initialize the model for training.
        Should set self.model with the initialized model.
        """
        pass

    @abstractmethod
    def train(
        self,
        num_epochs: int,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of training epochs
            output_dir: Directory to save checkpoints and logs
            resume_from_checkpoint: Optional path to checkpoint to resume from

        Returns:
            Dictionary containing training metrics and results
        """
        pass

    @abstractmethod
    def evaluate(self, data_path: str) -> Dict[str, float]:
        """
        Evaluate the model on given dataset.

        Args:
            data_path: Path to evaluation data

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        output_dir: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            output_dir: Directory to save checkpoint
            epoch: Current epoch number
            metrics: Optional metrics to save with checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint metadata
        """
        pass

    def get_model(self):
        """
        Get the underlying model.

        Returns:
            The trained model instance
        """
        return self.model

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics.

        Returns:
            Dictionary of training metrics
        """
        return self.metrics


class TrainerFactory:
    """Factory for creating trainer instances"""

    _trainers: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, name: str, trainer_class: Type[BaseTrainer]) -> None:
        """
        Register a trainer class.

        Args:
            name: Name to register the trainer under
            trainer_class: Trainer class to register
        """
        cls._trainers[name] = trainer_class
        logger.info(f"Registered trainer: {name}")

    @classmethod
    def create(
        cls,
        trainer_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseTrainer:
        """
        Create a trainer instance.

        Args:
            trainer_type: Type of trainer to create
            config: Configuration for the trainer

        Returns:
            Initialized trainer instance

        Raises:
            ValueError: If trainer_type is not registered
        """
        if trainer_type not in cls._trainers:
            available = ', '.join(cls._trainers.keys())
            raise ValueError(
                f"Unknown trainer type: {trainer_type}. Available: {available}"
            )

        trainer_class = cls._trainers[trainer_type]
        return trainer_class(config)

    @classmethod
    def get_available_trainers(cls) -> List[str]:
        """
        Get list of available trainers.

        Returns:
            List of registered trainer names
        """
        return list(cls._trainers.keys())


def register_trainer(name: str):
    """
    Decorator to register a trainer class.

    Args:
        name: Name to register the trainer under

    Returns:
        Decorator function

    Example:
        @register_trainer("hallucination")
        class HallucinationTrainer(BaseTrainer):
            ...
    """
    def decorator(trainer_class: Type[BaseTrainer]):
        TrainerFactory.register(name, trainer_class)
        return trainer_class
    return decorator
