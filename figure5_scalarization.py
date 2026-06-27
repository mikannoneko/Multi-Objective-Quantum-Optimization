"""Figure 5 多目标 scalarization 和目标预处理工具。

`w_ddts` 把三个真实目标转换为一个 DDTS 人工目标。论文中的
`wo_ddts` baseline 会分别训练三个 FM，因此 pipeline 使用
`compute_individual_objective_targets`，然后在 QUBO 层做 weighted sum。
"""

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
    """一次 scalarization 的结果和后续 checkpoint 需要的元数据。"""

    method: ScalarizationMethod
    setting: Figure5Setting
    objectives: tuple[str, ...]
    weights: tuple[float, ...]
    targets: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ObjectiveTargetsResult:
    """w/o DDTS 三个独立 FM 使用的方向一致 z-score targets。"""

    objectives: tuple[str, ...]
    targets: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def sample_preference_weights(rng: random.Random, num_objectives: int = 3) -> tuple[float, ...]:
    """采样非负且总和为 1 的 preference weights。"""

    if int(num_objectives) <= 0:
        raise ValueError("num_objectives must be positive")

    draws = [rng.expovariate(1.0) for _ in range(int(num_objectives))]
    total = sum(draws)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Unable to sample finite positive preference weights")
    return tuple(float(draw / total) for draw in draws)


def validate_preference_weights(weights: Sequence[float], num_objectives: int = 3) -> np.ndarray:
    """校验并轻微归一化权重，消除浮点求和误差。"""

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
    """构造 weighted-sum 数学参考目标。

    该函数保留用于单元测试和数学对照。论文 Figure 5 baseline 不使用该
    target 训练单个 FM，而是训练三个 FM 后合并 QUBO。
    """

    weight_array = validate_preference_weights(weights, num_objectives=len(FIGURE5_OBJECTIVES))
    individual = compute_individual_objective_targets(rows)
    target_matrix = np.column_stack([individual.targets[name] for name in FIGURE5_OBJECTIVES])
    targets = np.dot(target_matrix, weight_array).astype(np.float32)

    return ScalarizationResult(
        method="weighted_sum",
        setting="wo_ddts",
        objectives=FIGURE5_OBJECTIVES,
        weights=tuple(float(value) for value in weight_array),
        targets=targets,
        metadata={
            "objective_senses": dict(FIGURE5_OBJECTIVE_SENSES),
            **individual.metadata,
        },
    )


def compute_individual_objective_targets(rows: Sequence[Mapping[str, Any]]) -> ObjectiveTargetsResult:
    """Create three independently standardized minimization targets."""

    raw_values = _objective_matrix(rows)
    transformed = _to_minimization_values(raw_values)
    normalized, means, scales = _zscore_columns(transformed)
    return ObjectiveTargetsResult(
        objectives=FIGURE5_OBJECTIVES,
        targets={
            objective: normalized[:, index].astype(np.float32)
            for index, objective in enumerate(FIGURE5_OBJECTIVES)
        },
        metadata={
            "objective_senses": dict(FIGURE5_OBJECTIVE_SENSES),
            "transformed_objectives": {"kappa": "-kappa", "E": "-E", "rho": "rho"},
            "zscore_mean": _objective_metadata(means),
            "zscore_scale": _objective_metadata(scales),
        },
    )


def compute_ddts_targets(rows: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> ScalarizationResult:
    """构造 w/ DDTS 的 data-driven Tchebycheff 训练目标。

    按论文补充材料 S1.3，先对每个真实 objective 做 z-score，再在标准化
    空间中把 utopian point 放在当前最佳值前方 10%。输出仍为越小越好。
    """

    weight_array = validate_preference_weights(weights, num_objectives=len(FIGURE5_OBJECTIVES))
    raw_values = _objective_matrix(rows)
    normalized, means, scales = _zscore_columns(raw_values)
    utopian = np.array(
        [
            1.1 * np.max(normalized[:, 0]),
            1.1 * np.max(normalized[:, 1]),
            1.1 * np.min(normalized[:, 2]),
        ],
        dtype=np.float64,
    )
    distances = np.column_stack(
        (
            utopian[0] - normalized[:, 0],
            utopian[1] - normalized[:, 1],
            normalized[:, 2] - utopian[2],
        )
    )
    targets = np.max(distances * weight_array, axis=1).astype(np.float32)

    return ScalarizationResult(
        method="ddts",
        setting="w_ddts",
        objectives=FIGURE5_OBJECTIVES,
        weights=tuple(float(value) for value in weight_array),
        targets=targets,
        metadata={
            "objective_senses": dict(FIGURE5_OBJECTIVE_SENSES),
            "utopian_point": _objective_metadata(utopian),
            "utopian_space": "zscore",
            "zscore_mean": _objective_metadata(means),
            "zscore_scale": _objective_metadata(scales),
        },
    )


def scalarize_training_targets(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    setting: Figure5Setting,
) -> ScalarizationResult:
    """数学工具分发；pipeline 的 `wo_ddts` 会改用三 FM QUBO 合并。"""

    if setting == "wo_ddts":
        return compute_weighted_sum_targets(rows, weights)
    if setting == "w_ddts":
        return compute_ddts_targets(rows, weights)
    raise ValueError("setting must be 'wo_ddts' or 'w_ddts'")


def _objective_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """从 dataset rows 中抽取固定顺序的 `kappa/E/rho` 矩阵。"""

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
    """把 mixed-sense objectives 统一成最小化方向。"""

    return np.column_stack((-raw_values[:, 0], -raw_values[:, 1], raw_values[:, 2]))


def _zscore_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)
    scales = np.where(stds < SCALARIZATION_SCALE_FLOOR, 1.0, stds)
    return (values - means) / scales, means, scales


def _objective_metadata(values: Sequence[float]) -> Dict[str, float]:
    return {objective: float(value) for objective, value in zip(FIGURE5_OBJECTIVES, values)}


__all__ = [
    "FIGURE5_OBJECTIVES",
    "FIGURE5_OBJECTIVE_SENSES",
    "Figure5Setting",
    "ObjectiveTargetsResult",
    "ScalarizationMethod",
    "ScalarizationResult",
    "compute_ddts_targets",
    "compute_individual_objective_targets",
    "compute_weighted_sum_targets",
    "sample_preference_weights",
    "scalarize_training_targets",
    "validate_preference_weights",
]
