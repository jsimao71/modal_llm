"""Dependency-free task and calibration metrics."""

from __future__ import annotations

import math


def binary_auroc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def average_precision(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels), reverse=True)
    positives = sum(labels)
    if positives == 0:
        return math.nan
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def brier_score(labels: list[int], probabilities: list[float]) -> float:
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def expected_calibration_error(
    labels: list[int], probabilities: list[float], bins: int = 10
) -> float:
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            item for item, probability in enumerate(probabilities)
            if low <= probability < high or (index == bins - 1 and probability == 1.0)
        ]
        if selected:
            confidence = sum(probabilities[item] for item in selected) / len(selected)
            accuracy = sum(labels[item] for item in selected) / len(selected)
            error += len(selected) / len(labels) * abs(confidence - accuracy)
    return error
