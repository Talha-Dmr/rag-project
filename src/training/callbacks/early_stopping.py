"""
Early stopping callback to prevent overfitting.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to stop training when metric stops improving.

    Monitors a validation metric and stops training if it doesn't improve
    for a specified number of epochs (patience).
    """

    def __init__(
        self,
        patience: int = 3,
        metric_name: str = "val_f1_macro",
        mode: str = "max",
        min_delta: float = 0.0001,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs to wait for improvement
            metric_name: Name of metric to monitor
            mode: "max" to maximize metric, "min" to minimize
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to log messages
        """
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

        if verbose:
            logger.info(
                f"EarlyStopping: monitoring '{metric_name}' with patience={patience}"
            )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics
        """
        current_metric = metrics.get(self.metric_name)

        if current_metric is None:
            logger.warning(
                f"Metric '{self.metric_name}' not found in metrics, "
                "early stopping disabled"
            )
            return

        # Check if metric improved
        improved = False

        if self.mode == 'max':
            if current_metric > self.best_metric + self.min_delta:
                improved = True
        else:  # mode == 'min'
            if current_metric < self.best_metric - self.min_delta:
                improved = True

        if improved:
            self.best_metric = current_metric
            self.counter = 0
            self.best_epoch = epoch

            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: {self.metric_name} improved to "
                    f"{current_metric:.4f}"
                )
        else:
            self.counter += 1

            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: {self.metric_name} did not improve "
                    f"({current_metric:.4f} vs best {self.best_metric:.4f}). "
                    f"Patience: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.should_stop = True

                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best {self.metric_name}: {self.best_metric:.4f} "
                        f"at epoch {self.best_epoch + 1}"
                    )

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def stop_training(self) -> bool:
        """
        Check if training should be stopped.

        Returns:
            True if training should stop, False otherwise
        """
        return self.should_stop
