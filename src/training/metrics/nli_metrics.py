"""
Metrics computation for NLI classification tasks.

Provides accuracy, F1 scores, precision, recall, and confusion matrix
for 3-way classification (entailment/neutral/contradiction).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence, Any
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

    def __init__(self, calibration_bins: int = 15):
        """Initialize metrics tracker."""
        if calibration_bins <= 0:
            raise ValueError("calibration_bins must be a positive integer")
        self.calibration_bins = calibration_bins
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
        self._brier_total = 0.0
        self._brier_count = 0
        self._ece_bin_conf = [0.0 for _ in range(self.calibration_bins)]
        self._ece_bin_acc = [0.0 for _ in range(self.calibration_bins)]
        self._ece_bin_counts = [0 for _ in range(self.calibration_bins)]

    def update(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: Optional[Sequence[Sequence[float]]] = None
    ) -> None:
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: List of predicted labels (0, 1, or 2)
            labels: List of ground truth labels (0, 1, or 2)
            probabilities: Optional per-class probabilities aligned with predictions
        """
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)

        if probabilities is None:
            return

        if len(probabilities) != len(predictions):
            raise ValueError("probabilities length must match predictions length")

        for pred, label, probs in zip(predictions, labels, probabilities):
            if probs is None:
                continue
            prob_list = np.asarray(probs, dtype=float).tolist()
            if not prob_list:
                continue

            # Brier score (multi-class)
            brier = 0.0
            for idx, prob in enumerate(prob_list):
                target = 1.0 if idx == label else 0.0
                brier += (prob - target) ** 2
            self._brier_total += brier
            self._brier_count += 1

            # ECE binning using predicted class confidence
            if 0 <= pred < len(prob_list):
                conf = float(prob_list[pred])
                bin_idx = min(int(conf * self.calibration_bins), self.calibration_bins - 1)
                self._ece_bin_conf[bin_idx] += conf
                self._ece_bin_acc[bin_idx] += 1.0 if pred == label else 0.0
                self._ece_bin_counts[bin_idx] += 1

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
        class_labels = list(range(len(self.LABEL_NAMES)))
        f1_macro = f1_score(labels, predictions, average='macro', labels=class_labels, zero_division=0)
        precision_macro = precision_score(labels, predictions, average='macro', labels=class_labels, zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', labels=class_labels, zero_division=0)

        # Weighted-averaged metrics (weighted by support)
        f1_weighted = f1_score(labels, predictions, average='weighted', labels=class_labels, zero_division=0)
        precision_weighted = precision_score(labels, predictions, average='weighted', labels=class_labels, zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', labels=class_labels, zero_division=0)

        # Per-class metrics
        f1_per_class = f1_score(labels, predictions, average=None, labels=class_labels, zero_division=0)
        precision_per_class = precision_score(labels, predictions, average=None, labels=class_labels, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, labels=class_labels, zero_division=0)

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

        if self._brier_count > 0:
            metrics['brier'] = self._brier_total / self._brier_count

        total_calibration = sum(self._ece_bin_counts)
        if total_calibration > 0:
            ece = 0.0
            for idx in range(self.calibration_bins):
                count = self._ece_bin_counts[idx]
                if count == 0:
                    continue
                avg_conf = self._ece_bin_conf[idx] / count
                avg_acc = self._ece_bin_acc[idx] / count
                ece += (count / total_calibration) * abs(avg_acc - avg_conf)
            metrics['ece'] = ece

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
        if 'ece' in metrics:
            logger.info(f"  ECE: {metrics['ece']:.4f}")
        if 'brier' in metrics:
            logger.info(f"  Brier: {metrics['brier']:.4f}")

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
    labels: List[int],
    probabilities: Optional[Sequence[Sequence[float]]] = None
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
    metrics_tracker.update(predictions, labels, probabilities=probabilities)
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
