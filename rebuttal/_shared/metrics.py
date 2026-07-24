"""Metrics and small paper-output helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], num_classes: int
) -> Dict[str, object]:
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    class_ids = list(range(num_classes))
    report = classification_report(
        labels_array,
        predictions_array,
        labels=class_ids,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "macro_f1": float(
            f1_score(
                labels_array,
                predictions_array,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_array, predictions_array)
        ),
        "per_class": {str(index): report[str(index)] for index in class_ids},
        "confusion_matrix": confusion_matrix(
            labels_array, predictions_array, labels=class_ids
        ).tolist(),
        "predicted_class_counts": {
            str(index): int((predictions_array == index).sum()) for index in class_ids
        },
    }


def mean_std(values: Iterable[float]) -> Dict[str, object]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else float("nan"),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": array.tolist(),
    }


def aggregate_runs(
    runs: Sequence[Dict[str, object]], group_keys: Sequence[str]
) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for run in runs:
        key = tuple(run[field] for field in group_keys)
        groups.setdefault(key, []).append(run)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        row = {field: value for field, value in zip(group_keys, key)}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "train_time"):
            values = [float(item[metric]) for item in group if metric in item]
            if values:
                stats = mean_std(values)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
        row["seeds"] = ",".join(str(item["seed"]) for item in group)
        output.append(row)
    return output

