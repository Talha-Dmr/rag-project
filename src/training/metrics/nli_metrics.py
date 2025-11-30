"""
Metrics computation for NLI classification tasks.

Provides accuracy, F1 scores, precision, recall, and confusion matrix
for 3-way classification (entailment/neutral/contradiction).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


class NLIMetrics:
    """Compute and track NLI classification metrics."""

    LABEL_NAMES = ['entailment', 'neutral', 'contradiction']

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated predictions and labels."""
        self.all_predictions = []
        self.all_labels = []

    def update(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> None:
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: List of predicted labels (0, 1, or 2)
            labels: List of ground truth labels (0, 1, or 2)
        """
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)

    def compute(self, reset: bool = False) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.

        Args:
            reset: Whether to reset accumulated data after computation

        Returns:
            Dictionary containing all computed metrics
        """
        if not self.all_predictions:
            logger.warning("No predictions to compute metrics")
            return {}

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # Overall metrics
        accuracy = accuracy_score(labels, predictions)

        # Macro-averaged metrics (average across classes)
        f1_macro = f1_score(labels, predictions, average='macro')
        precision_macro = precision_score(labels, predictions, average='macro')
        recall_macro = recall_score(labels, predictions, average='macro')

        # Weighted-averaged metrics (weighted by support)
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_weighted = precision_score(labels, predictions, average='weighted')
        recall_weighted = recall_score(labels, predictions, average='weighted')

        # Per-class metrics
        f1_per_class = f1_score(labels, predictions, average=None)
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
        }

        # Add per-class metrics
        for i, label_name in enumerate(self.LABEL_NAMES):
            metrics[f'f1_{label_name}'] = f1_per_class[i]
            metrics[f'precision_{label_name}'] = precision_per_class[i]
            metrics[f'recall_{label_name}'] = recall_per_class[i]

        if reset:
            self.reset()

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.

        Returns:
            Confusion matrix of shape (3, 3)
        """
        if not self.all_predictions:
            return np.zeros((3, 3))

        return confusion_matrix(
            self.all_labels,
            self.all_predictions,
            labels=[0, 1, 2]
        )

    def get_classification_report(self) -> str:
        """
        Get detailed classification report.

        Returns:
            String containing classification report
        """
        if not self.all_predictions:
            return "No predictions available"

        return classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=self.LABEL_NAMES,
            digits=4
        )

    def log_metrics(self, prefix: str = "") -> None:
        """
        Log computed metrics.

        Args:
            prefix: Prefix to add to log messages (e.g., "Train" or "Val")
        """
        metrics = self.compute(reset=False)

        if not metrics:
            return

        prefix = f"{prefix} " if prefix else ""

        logger.info(f"{prefix}Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")

        logger.info(f"\n{prefix}Per-class metrics:")
        for label_name in self.LABEL_NAMES:
            logger.info(
                f"  {label_name}: "
                f"F1={metrics[f'f1_{label_name}']:.4f}, "
                f"P={metrics[f'precision_{label_name}']:.4f}, "
                f"R={metrics[f'recall_{label_name}']:.4f}"
            )


def compute_metrics_from_predictions(
    predictions: List[int],
    labels: List[int]
) -> Dict[str, float]:
    """
    Compute metrics from predictions and labels (one-shot).

    Args:
        predictions: List of predicted labels
        labels: List of ground truth labels

    Returns:
        Dictionary of computed metrics
    """
    metrics_tracker = NLIMetrics()
    metrics_tracker.update(predictions, labels)
    return metrics_tracker.compute()


def print_confusion_matrix(
    cm: np.ndarray,
    label_names: Optional[List[str]] = None
) -> None:
    """
    Pretty print confusion matrix.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes)
        label_names: Optional list of label names
    """
    if label_names is None:
        label_names = NLIMetrics.LABEL_NAMES

    # Header
    header = "Confusion Matrix:\n"
    header += "Predicted →\n"
    header += "Actual ↓      "
    for name in label_names:
        header += f"{name[:12]:>12}  "
    header += "\n"

    print(header)

    # Rows
    for i, actual in enumerate(label_names):
        row = f"{actual[:12]:>12}  "
        for j in range(len(label_names)):
            row += f"{cm[i, j]:>12}  "
        print(row)

    # Total
    total = cm.sum()
    correct = cm.diagonal().sum()
    accuracy = correct / total if total > 0 else 0
    print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")
