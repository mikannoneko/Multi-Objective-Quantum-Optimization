from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Sequence

import numpy as np


FIGURE5_OBJECTIVES = ("kappa", "E", "rho")
FIGURE5_OBJECTIVE_SENSES = {"kappa": "maximize", "E": "maximize", "rho": "minimize"}
WEIGHT_SUM_TOLERANCE = 1e-10
SCALARIZATION_SCALE_FLOOR = 1e-12

Figure5Setting = Literal["w_ddts", "wo_ddts"]
ScalarizationMethod = Literal["weighted_sum", "ddts"]


@dataclass(frozen=True)
class ScalarizationResult:
    method: ScalarizationMethod
    setting: Figure5Setting
    objectives: tuple[str, ...]
    weights: tuple[float, ...]
    targets: np.ndarray
    metadata: Dict[str, Any]


def sample_preference_weights(rng: random.Random, num_objectives: int = 3) -> tuple[float, ...]:
    if int(num_objectives) <= 0:
        raise ValueError("num_objectives must be positive")

    draws = [rng.expovariate(1.0) for _ in range(int(num_objectives))]
    total = sum(draws)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Unable to sample finite positive preference weights")
    return tuple(float(draw / total) for draw in draws)


def validate_preference_weights(weights: Sequence[float], num_objectives: int = 3) -> np.ndarray:
    if len(weights) != int(num_objectives):
        raise ValueError(f"weights must contain {num_objectives} values")

    array = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must be finite")
    if np.any(array < 0.0):
        raise ValueError("weights must be non-negative")

    total = float(np.sum(array))
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got {total}")
    return array / total


def compute_weighted_sum_targets(rows: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> ScalarizationResult:
    weight_array = validate_preference_weights(weights, num_objectives=len(FIGURE5_OBJECTIVES))
    raw_values = _objective_matrix(rows)
    transformed = _to_minimization_values(raw_values)
    means = np.mean(transformed, axis=0)
    stds = np.std(transformed, axis=0)
    scales = np.where(stds < SCALARIZATION_SCALE_FLOOR, 1.0, stds)
    normalized = (transformed - means) / scales
    targets = np.dot(normalized, weight_array).astype(np.float32)

    return ScalarizationResult(
        method="weighted_sum",
        setting="wo_ddts",
        objectives=FIGURE5_OBJECTIVES,
        weights=tuple(float(value) for value in weight_array),
        targets=targets,
        metadata={
            "objective_senses": dict(FIGURE5_OBJECTIVE_SENSES),
            "transformed_objectives": {"kappa": "-kappa", "E": "-E", "rho": "rho"},
            "zscore_mean": _objective_metadata(means),
            "zscore_scale": _objective_metadata(scales),
        },
    )


def compute_ddts_targets(rows: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> ScalarizationResult:
    weight_array = validate_preference_weights(weights, num_objectives=len(FIGURE5_OBJECTIVES))
    raw_values = _objective_matrix(rows)
    utopian = np.array(
        [
            np.max(raw_values[:, 0]),
            np.max(raw_values[:, 1]),
            np.min(raw_values[:, 2]),
        ],
        dtype=np.float64,
    )
    distances = np.column_stack(
        (
            utopian[0] - raw_values[:, 0],
            utopian[1] - raw_values[:, 1],
            raw_values[:, 2] - utopian[2],
        )
    )
    ranges = np.array(
        [
            np.max(raw_values[:, 0]) - np.min(raw_values[:, 0]),
            np.max(raw_values[:, 1]) - np.min(raw_values[:, 1]),
            np.max(raw_values[:, 2]) - np.min(raw_values[:, 2]),
        ],
        dtype=np.float64,
    )
    scales = np.where(ranges < SCALARIZATION_SCALE_FLOOR, 1.0, ranges)
    normalized_distances = distances / scales
    targets = np.max(normalized_distances * weight_array, axis=1).astype(np.float32)

    return ScalarizationResult(
        method="ddts",
        setting="w_ddts",
        objectives=FIGURE5_OBJECTIVES,
        weights=tuple(float(value) for value in weight_array),
        targets=targets,
        metadata={
            "objective_senses": dict(FIGURE5_OBJECTIVE_SENSES),
            "utopian_point": _objective_metadata(utopian),
            "range_scale": _objective_metadata(scales),
        },
    )


def scalarize_training_targets(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    setting: Figure5Setting,
) -> ScalarizationResult:
    if setting == "wo_ddts":
        return compute_weighted_sum_targets(rows, weights)
    if setting == "w_ddts":
        return compute_ddts_targets(rows, weights)
    raise ValueError("setting must be 'wo_ddts' or 'w_ddts'")


def _objective_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("rows must not be empty")

    matrix = np.empty((len(rows), len(FIGURE5_OBJECTIVES)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for objective_index, objective in enumerate(FIGURE5_OBJECTIVES):
            if objective not in row:
                raise ValueError(f"row {row_index} missing objective {objective!r}")
            value = float(row[objective])
            if not math.isfinite(value):
                raise ValueError(f"row {row_index} objective {objective!r} must be finite")
            matrix[row_index, objective_index] = value
    return matrix


def _to_minimization_values(raw_values: np.ndarray) -> np.ndarray:
    return np.column_stack((-raw_values[:, 0], -raw_values[:, 1], raw_values[:, 2]))


def _objective_metadata(values: Sequence[float]) -> Dict[str, float]:
    return {objective: float(value) for objective, value in zip(FIGURE5_OBJECTIVES, values)}


__all__ = [
    "FIGURE5_OBJECTIVES",
    "FIGURE5_OBJECTIVE_SENSES",
    "Figure5Setting",
    "ScalarizationMethod",
    "ScalarizationResult",
    "compute_ddts_targets",
    "compute_weighted_sum_targets",
    "sample_preference_weights",
    "scalarize_training_targets",
    "validate_preference_weights",
]
