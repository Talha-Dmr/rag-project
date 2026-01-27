import pytest

from src.training.metrics.nli_metrics import NLIMetrics


def test_ece_brier_perfect_predictions():
    metrics = NLIMetrics(calibration_bins=10)
    predictions = [0, 1]
    labels = [0, 1]
    probabilities = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    metrics.update(predictions, labels, probabilities=probabilities)
    results = metrics.compute()

    assert results["ece"] == pytest.approx(0.0)
    assert results["brier"] == pytest.approx(0.0)


def test_ece_brier_with_overconfident_probs():
    metrics = NLIMetrics(calibration_bins=1)
    predictions = [0, 1]
    labels = [0, 1]
    probabilities = [
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
    ]

    metrics.update(predictions, labels, probabilities=probabilities)
    results = metrics.compute()

    assert results["ece"] == pytest.approx(0.4)
    assert results["brier"] == pytest.approx(0.24)
