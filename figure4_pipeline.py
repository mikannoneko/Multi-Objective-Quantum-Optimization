"""Figure 4 active-learning 主流程。

一条 trajectory 对应固定的 `setting + objective + seed`：从同一个初始数据集出发，
重复训练 FM、转 QUBO、从 SA reads 选择可行候选、处理重复 composition，并记录
该 objective 的 best-so-far 曲线。
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

from alloy_dataset_generator import (
    CSV_FIELDNAMES,
    build_dataset_row,
    generate_initial_dataset_batch,
    sample_single_objective_design,
)
from figure4_experiment_config import (
    FIGURE4_OBJECTIVE_NAMES,
    FIGURE4_OBJECTIVES,
    FIGURE4_SETTINGS,
    Figure4ExperimentConfig,
    Figure4Setting,
    ObjectiveSpec,
)
from fm_torch import fit_torch_fm, fm_to_qubo
from figure4_outputs import (
    SUMMARY_SCHEMA_VERSION,
    checkpoint_payload,
    figure4_output_layout,
    load_checkpoint,
    write_checkpoint,
)
from qubo_math import (
    IterationEncoding,
    ONE_HOT_PENALTY_WEIGHT,
    QuboBuildResult,
    QuboStats,
    SYSTEM_PENALTY_WEIGHT,
    build_single_objective_qubo,
    prepare_discrete_composition,
    select_lowest_energy_feasible_sample,
    solve_qubo_with_sa,
    validate_candidate_composition,
)
from figure4_setting_strategies import (
    SettingStrategy,
    get_setting_strategy,
    validate_settings,
)
from experiment_runtime import (
    FIGURE4_REPLACEMENT_NAMESPACE,
    FIGURE4_SEED_INDEX,
    FM_TRAINING_BACKEND,
    MAX_RANDOM_REPLACEMENT_ATTEMPTS,
    SEED_DERIVATION_SCHEME,
    derive_bounded_seed,
    derive_fm_seed_root,
    derive_python_seed,
    fm_seed_block_size,
    fm_seed_plan,
    require_integer,
    same_json_value,
    validate_seed_schedule,
    write_json_atomic,
)


TRAINING_BACKEND = FM_TRAINING_BACKEND
CandidateStatus = Literal["accepted", "duplicate_replacement"]
LOGGER = logging.getLogger(__name__)
_STATE_COUNTER_FIELDS = (
    "duplicate_replacements",
    "random_replacements",
    "random_replacement_draws",
    "accepted_sa_candidates",
    "infeasible_sa_samples_skipped",
    "max_feasible_candidate_rank",
)
_QUBO_FLOAT_FIELDS = (
    "fm_scale",
    "system_scale",
    "one_hot_scale",
    "system_penalty_weight",
    "one_hot_penalty_weight",
    "max_abs",
)
_DATASET_ROW_FIELDS = frozenset(CSV_FIELDNAMES)
_DATASET_NUMERIC_FIELDS = tuple(name for name in CSV_FIELDNAMES if name not in {"sample_id", "seed"})
FIGURE4_TRAJECTORY_COUNT = len(FIGURE4_OBJECTIVES) * len(FIGURE4_SETTINGS)
FIGURE4_BOUNDED_STREAM_COUNT = 2
FIGURE4_CGFM_STREAM_INDEX = 0
FIGURE4_SA_STREAM_INDEX = 1


def _trajectory_index(objective: ObjectiveSpec, setting: Figure4Setting) -> int:
    try:
        objective_index = FIGURE4_OBJECTIVE_NAMES.index(objective.name)
        setting_index = FIGURE4_SETTINGS.index(setting)
    except ValueError as exc:
        raise ValueError(
            f"Cannot derive Figure 4 trajectory seed for objective={objective.name!r}, setting={setting!r}"
        ) from exc
    return (objective_index * len(FIGURE4_SETTINGS)) + setting_index


def _validate_seed_schedule(
    seed_list: Sequence[int], config: Figure4ExperimentConfig
) -> list[int]:
    return validate_seed_schedule(
        seed_list,
        figure_index=FIGURE4_SEED_INDEX,
        trajectory_count=FIGURE4_TRAJECTORY_COUNT,
        iterations=config.iterations,
        bounded_stream_count=FIGURE4_BOUNDED_STREAM_COUNT,
        fm_model_count=1,
        fm_seed_block_size=fm_seed_block_size(config.optuna_trials),
    )


@dataclass
class TrajectoryState:
    """可写入 checkpoint 的运行中状态。"""

    rows: List[Dict[str, Any]]
    best_so_far: List[float] = field(default_factory=list)
    fm_metadata: Dict[str, Any] = field(default_factory=dict)
    qubo_stats: QuboStats | None = None
    duplicate_replacements: int = 0
    random_replacements: int = 0
    random_replacement_draws: int = 0
    accepted_sa_candidates: int = 0
    infeasible_sa_samples_skipped: int = 0
    max_feasible_candidate_rank: int = 0

    @property
    def completed_iterations(self) -> int:
        return len(self.best_so_far)


@dataclass(frozen=True)
class TrajectoryResult:
    """一条完整 trajectory 的对外结果，用于 summary 和跨 seed 聚合。"""

    objective: str
    setting: str
    seed: int
    best_so_far: List[float]
    final_dataset_size: int
    training_backend: str
    fm_metadata: Dict[str, Any]
    qubo_stats: QuboStats | None = None
    duplicate_replacements: int = 0
    random_replacements: int = 0
    random_replacement_draws: int = 0
    accepted_sa_candidates: int = 0
    infeasible_sa_samples_skipped: int = 0
    max_feasible_candidate_rank: int = 0
    completed_iterations: int = 0


@dataclass(frozen=True)
class CandidateDecision:
    status: CandidateStatus
    row: Dict[str, Any]
    replacement_draws: int = 0


@dataclass(frozen=True)
class TrainingIterationResult:
    encoding: IterationEncoding
    candidate_bits: np.ndarray
    feasible_candidate_rank: int
    infeasible_sa_samples_skipped: int
    fm_metadata: Dict[str, Any]
    qubo_stats: QuboStats


@dataclass(frozen=True)
class AggregatedTrajectory:
    objective: str
    setting: str
    best_so_far_mean: List[float]
    best_so_far_std: List[float]
    best_so_far_min: List[float]
    best_so_far_max: List[float]
    num_trajectories: int


@dataclass(frozen=True)
class Figure4Summary:
    schema_version: int
    seed_derivation: str
    training_backend: str
    config: Dict[str, Any]
    seed_list: List[int]
    objectives: List[str]
    settings: List[str]
    trajectories: List[TrajectoryResult]
    aggregated: Dict[str, AggregatedTrajectory]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalized_composition_from_row(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
        dtype=np.float64,
    )


def zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        std = 1.0
    return ((values - mean) / std).astype(np.float32), mean, std


def discretize_row_for_figure4(row: Dict[str, Any], num_levels: int) -> Dict[str, Any]:
    """把连续初始样本固化到当前 one-hot level 网格上，再重算真实物性。"""

    discrete_composition = prepare_discrete_composition(normalized_composition_from_row(row), num_levels)
    return build_dataset_row(row["sample_id"], row["seed"], discrete_composition)


def discretize_rows_for_figure4(rows: Sequence[Dict[str, Any]], num_levels: int) -> List[Dict[str, Any]]:
    return [discretize_row_for_figure4(row, num_levels) for row in rows]


def _sample_random_replacement_row(
    sample_id: int,
    seed: int,
    rng: random.Random,
    num_levels: int,
) -> Dict[str, Any]:
    """候选与已有 composition 重复时生成同一离散网格上的替代样本。"""

    continuous_design = sample_single_objective_design(rng)
    discrete_composition = prepare_discrete_composition(continuous_design, num_levels)
    return build_dataset_row(sample_id, seed, discrete_composition)


def _composition_key(composition: Sequence[float]) -> Tuple[float, float, float, float]:
    return tuple(round(float(value), 10) for value in composition)


def _row_composition_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return _composition_key(normalized_composition_from_row(row))


def _checkpoint_state_error(checkpoint: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid checkpoint state {checkpoint}: {detail}")


def _is_json_number(value: Any) -> bool:
    return type(value) in {int, float}


def _state_from_checkpoint(payload: Dict[str, Any], checkpoint: Path) -> TrajectoryState:
    """严格反序列化 schema v4 state，不为缺失字段提供默认值。"""

    raw_state = payload["state"]
    expected_fields = {item.name for item in dataclass_fields(TrajectoryState)}
    missing_fields = sorted(expected_fields - raw_state.keys())
    unexpected_fields = sorted(raw_state.keys() - expected_fields)
    if missing_fields:
        raise _checkpoint_state_error(checkpoint, f"missing fields: {', '.join(missing_fields)}")
    if unexpected_fields:
        raise _checkpoint_state_error(checkpoint, f"unexpected fields: {', '.join(unexpected_fields)}")

    if not isinstance(raw_state["rows"], list) or any(not isinstance(row, dict) for row in raw_state["rows"]):
        raise _checkpoint_state_error(checkpoint, "field 'rows' must be a list of JSON objects")
    if not isinstance(raw_state["best_so_far"], list) or any(
        not _is_json_number(value) for value in raw_state["best_so_far"]
    ):
        raise _checkpoint_state_error(checkpoint, "field 'best_so_far' must be a list of numbers")
    if not isinstance(raw_state["fm_metadata"], dict):
        raise _checkpoint_state_error(checkpoint, "field 'fm_metadata' must be a JSON object")
    for counter_name in _STATE_COUNTER_FIELDS:
        if type(raw_state[counter_name]) is not int:
            raise _checkpoint_state_error(checkpoint, f"field {counter_name!r} must be an integer")

    raw_qubo_stats = raw_state["qubo_stats"]
    if raw_qubo_stats is None:
        qubo_stats = None
    else:
        if not isinstance(raw_qubo_stats, dict):
            raise _checkpoint_state_error(checkpoint, "field 'qubo_stats' must be null or a JSON object")
        expected_qubo_fields = {item.name for item in dataclass_fields(QuboStats)}
        missing_qubo_fields = sorted(expected_qubo_fields - raw_qubo_stats.keys())
        unexpected_qubo_fields = sorted(raw_qubo_stats.keys() - expected_qubo_fields)
        if missing_qubo_fields:
            raise _checkpoint_state_error(
                checkpoint,
                f"field 'qubo_stats' is missing: {', '.join(missing_qubo_fields)}",
            )
        if unexpected_qubo_fields:
            raise _checkpoint_state_error(
                checkpoint,
                f"field 'qubo_stats' has unexpected fields: {', '.join(unexpected_qubo_fields)}",
            )
        for field_name in _QUBO_FLOAT_FIELDS:
            if not _is_json_number(raw_qubo_stats[field_name]):
                raise _checkpoint_state_error(
                    checkpoint,
                    f"field 'qubo_stats.{field_name}' must be a number",
                )
        if type(raw_qubo_stats["num_variables"]) is not int:
            raise _checkpoint_state_error(checkpoint, "field 'qubo_stats.num_variables' must be an integer")
        qubo_stats = QuboStats(
            fm_scale=float(raw_qubo_stats["fm_scale"]),
            system_scale=float(raw_qubo_stats["system_scale"]),
            one_hot_scale=float(raw_qubo_stats["one_hot_scale"]),
            system_penalty_weight=float(raw_qubo_stats["system_penalty_weight"]),
            one_hot_penalty_weight=float(raw_qubo_stats["one_hot_penalty_weight"]),
            num_variables=int(raw_qubo_stats["num_variables"]),
            max_abs=float(raw_qubo_stats["max_abs"]),
        )

    return TrajectoryState(
        rows=[dict(row) for row in raw_state["rows"]],
        best_so_far=[float(value) for value in raw_state["best_so_far"]],
        fm_metadata=dict(raw_state["fm_metadata"]),
        qubo_stats=qubo_stats,
        duplicate_replacements=raw_state["duplicate_replacements"],
        random_replacements=raw_state["random_replacements"],
        random_replacement_draws=raw_state["random_replacement_draws"],
        accepted_sa_candidates=raw_state["accepted_sa_candidates"],
        infeasible_sa_samples_skipped=raw_state["infeasible_sa_samples_skipped"],
        max_feasible_candidate_rank=raw_state["max_feasible_candidate_rank"],
    )


def _validate_trajectory_state(
    state: TrajectoryState,
    *,
    objective: ObjectiveSpec,
    setting: Figure4Setting,
    seed: int,
    config: Figure4ExperimentConfig,
    checkpoint: Path,
) -> None:
    """验证恢复状态的行、曲线、计数和最后一轮 QUBO 元数据彼此一致。"""

    completed = state.completed_iterations
    if completed > config.iterations:
        raise _checkpoint_state_error(
            checkpoint,
            f"completed_iterations={completed} exceeds configured iterations={config.iterations}",
        )

    expected_row_count = config.num_samples + completed
    if len(state.rows) != expected_row_count:
        raise _checkpoint_state_error(
            checkpoint,
            f"rows has length {len(state.rows)}; expected num_samples + completed_iterations = {expected_row_count}",
        )

    for row_index, row in enumerate(state.rows):
        missing_row_fields = sorted(_DATASET_ROW_FIELDS - row.keys())
        if missing_row_fields:
            raise _checkpoint_state_error(
                checkpoint,
                f"rows[{row_index}] is missing fields: {', '.join(missing_row_fields)}",
            )
        if type(row["sample_id"]) is not int or row["sample_id"] != row_index:
            raise _checkpoint_state_error(
                checkpoint,
                f"rows[{row_index}].sample_id must equal its row index {row_index}",
            )
        if type(row["seed"]) is not int or row["seed"] != int(seed):
            raise _checkpoint_state_error(
                checkpoint,
                f"rows[{row_index}].seed must equal trajectory seed {int(seed)}",
            )
        for field_name in _DATASET_NUMERIC_FIELDS:
            value = row[field_name]
            if not _is_json_number(value) or not math.isfinite(float(value)):
                raise _checkpoint_state_error(
                    checkpoint,
                    f"rows[{row_index}].{field_name} must be a finite number",
                )
        composition = normalized_composition_from_row(row)
        if np.any(composition < 0.0) or abs(float(np.sum(composition)) - 1.0) > 1e-10:
            raise _checkpoint_state_error(
                checkpoint,
                f"rows[{row_index}] composition must be non-negative and sum to 1",
            )

    expected_best = objective.initial_best()
    for row in state.rows[: config.num_samples]:
        expected_best = objective.update_best(expected_best, float(row[objective.name]))
    for iteration, actual_best in enumerate(state.best_so_far):
        if not math.isfinite(actual_best):
            raise _checkpoint_state_error(
                checkpoint,
                f"best_so_far[{iteration}] must be finite",
            )
        added_row = state.rows[config.num_samples + iteration]
        expected_best = objective.update_best(expected_best, float(added_row[objective.name]))
        if not math.isclose(actual_best, expected_best, rel_tol=1e-12, abs_tol=1e-12):
            raise _checkpoint_state_error(
                checkpoint,
                f"best_so_far[{iteration}]={actual_best!r} does not match rows-derived value {expected_best!r}",
            )

    for counter_name in _STATE_COUNTER_FIELDS:
        if getattr(state, counter_name) < 0:
            raise _checkpoint_state_error(checkpoint, f"counter {counter_name!r} must be non-negative")
    if state.accepted_sa_candidates + state.duplicate_replacements != completed:
        raise _checkpoint_state_error(
            checkpoint,
            "accepted_sa_candidates + duplicate_replacements must equal completed_iterations",
        )
    if state.random_replacements != state.duplicate_replacements:
        raise _checkpoint_state_error(
            checkpoint,
            "random_replacements must equal duplicate_replacements",
        )
    if state.random_replacement_draws < state.random_replacements:
        raise _checkpoint_state_error(
            checkpoint,
            "random_replacement_draws must be at least random_replacements",
        )
    maximum_replacement_draws = state.duplicate_replacements * MAX_RANDOM_REPLACEMENT_ATTEMPTS
    if state.random_replacement_draws > maximum_replacement_draws:
        raise _checkpoint_state_error(
            checkpoint,
            f"random_replacement_draws exceeds the maximum {maximum_replacement_draws}",
        )

    if completed == 0:
        if state.infeasible_sa_samples_skipped != 0 or state.max_feasible_candidate_rank != 0:
            raise _checkpoint_state_error(checkpoint, "zero-iteration state must have zero SA audit counters")
        if state.qubo_stats is not None or state.fm_metadata:
            raise _checkpoint_state_error(checkpoint, "zero-iteration state must not contain training metadata")
        return

    if not 1 <= state.max_feasible_candidate_rank <= config.sa_reads:
        raise _checkpoint_state_error(
            checkpoint,
            f"max_feasible_candidate_rank must be between 1 and sa_reads={config.sa_reads}",
        )
    minimum_skipped = state.max_feasible_candidate_rank - 1
    maximum_skipped = completed * (config.sa_reads - 1)
    if not minimum_skipped <= state.infeasible_sa_samples_skipped <= maximum_skipped:
        raise _checkpoint_state_error(
            checkpoint,
            "infeasible_sa_samples_skipped is inconsistent with completed iterations and feasible rank",
        )

    trajectory_index = _trajectory_index(objective, setting)
    expected_fm_root = derive_fm_seed_root(
        seed,
        trajectory_count=FIGURE4_TRAJECTORY_COUNT,
        trajectory_index=trajectory_index,
        iterations=config.iterations,
        iteration=completed - 1,
        model_count=1,
        model_index=0,
        figure_index=FIGURE4_SEED_INDEX,
        block_size=fm_seed_block_size(config.optuna_trials),
    )
    expected_seed_plan = fm_seed_plan(expected_fm_root, config.optuna_trials)
    if "seed_plan" not in state.fm_metadata:
        raise _checkpoint_state_error(checkpoint, "fm_metadata is missing field 'seed_plan'")
    if not same_json_value(state.fm_metadata["seed_plan"], expected_seed_plan):
        raise _checkpoint_state_error(
            checkpoint,
            "fm_metadata.seed_plan does not match the derived latest-iteration FM schedule",
        )

    stats = state.qubo_stats
    if stats is None:
        raise _checkpoint_state_error(checkpoint, "completed state must contain qubo_stats")
    for field_name in _QUBO_FLOAT_FIELDS:
        value = float(getattr(stats, field_name))
        if not math.isfinite(value) or value < 0.0:
            raise _checkpoint_state_error(
                checkpoint,
                f"qubo_stats.{field_name} must be finite and non-negative",
            )
    if stats.fm_scale <= 0.0 or stats.one_hot_scale <= 0.0:
        raise _checkpoint_state_error(
            checkpoint,
            "qubo_stats FM and one-hot normalization scales must be positive",
        )
    expected_num_variables = (4 if setting == "wo_cgfm" else 3) * config.num_levels
    if stats.num_variables != expected_num_variables:
        raise _checkpoint_state_error(
            checkpoint,
            f"qubo_stats.num_variables={stats.num_variables}; expected {expected_num_variables} for {setting}",
        )
    if not math.isclose(stats.one_hot_penalty_weight, ONE_HOT_PENALTY_WEIGHT, abs_tol=1e-12):
        raise _checkpoint_state_error(checkpoint, "qubo_stats.one_hot_penalty_weight is inconsistent")
    expected_system_weight = SYSTEM_PENALTY_WEIGHT if setting == "wo_cgfm" else 0.0
    if not math.isclose(stats.system_penalty_weight, expected_system_weight, abs_tol=1e-12):
        raise _checkpoint_state_error(checkpoint, "qubo_stats.system_penalty_weight is inconsistent with setting")
    if setting == "wo_cgfm" and stats.system_scale <= 0.0:
        raise _checkpoint_state_error(checkpoint, "wo_cgfm qubo_stats.system_scale must be positive")
    if setting == "w_cgfm" and not math.isclose(stats.system_scale, 0.0, abs_tol=1e-12):
        raise _checkpoint_state_error(checkpoint, "w_cgfm qubo_stats.system_scale must be zero")


def _initial_state(
    initial_rows: Sequence[Dict[str, Any]], config: Figure4ExperimentConfig
) -> TrajectoryState:
    return TrajectoryState(rows=discretize_rows_for_figure4(initial_rows, config.num_levels))


def _current_best_value(state: TrajectoryState, objective: ObjectiveSpec) -> float:
    if state.best_so_far:
        return state.best_so_far[-1]
    best_value = objective.initial_best()
    for row in state.rows:
        best_value = objective.update_best(best_value, float(row[objective.name]))
    return best_value


def _sample_unique_replacement_row(
    *,
    sample_id: int,
    seed: int,
    rng: random.Random,
    num_levels: int,
    seen_compositions: set[Tuple[float, float, float, float]],
    max_attempts: int = MAX_RANDOM_REPLACEMENT_ATTEMPTS,
) -> tuple[Dict[str, Any], int]:
    for attempt in range(1, max_attempts + 1):
        row = _sample_random_replacement_row(sample_id, seed, rng, num_levels)
        if _row_composition_key(row) not in seen_compositions:
            return row, attempt
    raise RuntimeError(f"Unable to generate a novel random replacement after {max_attempts} attempts")


def _fit_and_solve_iteration(
    *,
    rows: Sequence[Dict[str, Any]],
    objective: ObjectiveSpec,
    strategy: SettingStrategy,
    config: Figure4ExperimentConfig,
    seed: int,
    iteration: int,
) -> TrainingIterationResult:
    """完成单轮“当前数据集 -> FM -> QUBO -> SA 候选 bit vector”。"""

    trajectory_index = _trajectory_index(objective, strategy.name)
    encoding_seed = derive_bounded_seed(
        seed,
        trajectory_count=FIGURE4_TRAJECTORY_COUNT,
        trajectory_index=trajectory_index,
        iterations=config.iterations,
        iteration=iteration,
        stream_count=FIGURE4_BOUNDED_STREAM_COUNT,
        stream_index=FIGURE4_CGFM_STREAM_INDEX,
        figure_index=FIGURE4_SEED_INDEX,
    )
    fm_seed_root = derive_fm_seed_root(
        seed,
        trajectory_count=FIGURE4_TRAJECTORY_COUNT,
        trajectory_index=trajectory_index,
        iterations=config.iterations,
        iteration=iteration,
        model_count=1,
        model_index=0,
        figure_index=FIGURE4_SEED_INDEX,
        block_size=fm_seed_block_size(config.optuna_trials),
    )
    sa_seed = derive_bounded_seed(
        seed,
        trajectory_count=FIGURE4_TRAJECTORY_COUNT,
        trajectory_index=trajectory_index,
        iterations=config.iterations,
        iteration=iteration,
        stream_count=FIGURE4_BOUNDED_STREAM_COUNT,
        stream_index=FIGURE4_SA_STREAM_INDEX,
        figure_index=FIGURE4_SEED_INDEX,
    )
    encoding = strategy.create_encoding(config.num_levels, encoding_seed)
    features = strategy.encode_rows(rows, encoding)
    raw_targets = np.array([float(row[objective.name]) for row in rows], dtype=np.float64)
    # 所有 objective 都转换成最小化方向后再 z-score；QUBO/SA 后续也是最小化。
    transformed_targets = objective.transform_for_training(raw_targets)
    scaled_targets, _, _ = zscore(transformed_targets)

    model, fm_metadata = fit_torch_fm(
        features,
        scaled_targets,
        optuna_trials=config.optuna_trials,
        device=config.device,
        seed=fm_seed_root,
    )
    fm_q, fm_bias = fm_to_qubo(model)
    qubo_result: QuboBuildResult = build_single_objective_qubo(
        fm_q,
        fm_bias,
        encoding,
        include_system_penalty=strategy.include_system_penalty,
    )
    sampling = solve_qubo_with_sa(
        qubo_result.q,
        qubo_result.bias,
        reads=config.sa_reads,
        sweeps=config.sa_sweeps,
        seed=sa_seed,
    )

    def is_feasible(candidate_bits: np.ndarray) -> bool:
        composition = strategy.decode_candidate(candidate_bits, encoding)
        return composition is not None and validate_candidate_composition(composition)

    try:
        selected = select_lowest_energy_feasible_sample(sampling, is_feasible)
    except RuntimeError as exc:
        one_hot_valid = 0
        fully_feasible = 0
        for sample in sampling.samples:
            composition = strategy.decode_candidate(sample.state, encoding)
            if composition is None:
                continue
            one_hot_valid += 1
            if validate_candidate_composition(composition):
                fully_feasible += 1
        detail = (
            "Figure 4 SA found no feasible candidate for "
            f"setting={strategy.name!r}, objective={objective.name!r}, seed={seed}, "
            f"iteration={iteration + 1}/{config.iterations}: "
            f"one_hot_valid={one_hot_valid}/{sampling.num_samples}, "
            f"fully_feasible={fully_feasible}/{sampling.num_samples}, "
            f"num_levels={config.num_levels}, sa_sweeps={config.sa_sweeps}. "
            "This is a deterministic heuristic-sampling failure; unchanged --resume "
            "repeats the same SA batch. Use a separate output directory with a smaller "
            "--num-levels or another SA configuration."
        )
        LOGGER.error(detail)
        raise RuntimeError(detail) from exc
    return TrainingIterationResult(
        encoding=encoding,
        candidate_bits=selected.state,
        feasible_candidate_rank=selected.rank,
        infeasible_sa_samples_skipped=selected.infeasible_samples_skipped,
        fm_metadata=dict(fm_metadata),
        qubo_stats=qubo_result.stats,
    )


def _decide_candidate(
    *,
    candidate_bits: np.ndarray,
    encoding: IterationEncoding,
    strategy: SettingStrategy,
    sample_id: int,
    seed: int,
    replacement_rng: random.Random,
    num_levels: int,
    seen_compositions: set[Tuple[float, float, float, float]],
) -> CandidateDecision:
    """校验已筛选的 SA 候选，并在 composition 重复时生成替代样本。"""

    candidate_composition = strategy.decode_candidate(candidate_bits, encoding)
    if candidate_composition is None or not validate_candidate_composition(candidate_composition):
        raise RuntimeError("Internal error: selected SA candidate is not feasible")

    if _composition_key(candidate_composition) in seen_compositions:
        # 重复 composition 也替换，避免 active-learning 数据集里出现同一点。
        row, draws = _sample_unique_replacement_row(
            sample_id=sample_id,
            seed=seed,
            rng=replacement_rng,
            num_levels=num_levels,
            seen_compositions=seen_compositions,
        )
        return CandidateDecision(status="duplicate_replacement", row=row, replacement_draws=draws)

    return CandidateDecision(status="accepted", row=build_dataset_row(sample_id, seed, candidate_composition))


def _apply_candidate_decision(state: TrajectoryState, decision: CandidateDecision) -> None:
    """更新 replacement/accepted 计数并把最终 row 加入 trajectory 数据集。"""

    if decision.status == "accepted":
        state.accepted_sa_candidates += 1
    elif decision.status == "duplicate_replacement":
        state.duplicate_replacements += 1
        state.random_replacements += 1
        state.random_replacement_draws += decision.replacement_draws
    else:
        raise ValueError(f"Unsupported candidate decision status: {decision.status}")
    state.rows.append(decision.row)


def _result_from_state(
    *,
    objective: ObjectiveSpec,
    setting: Figure4Setting,
    seed: int,
    state: TrajectoryState,
) -> TrajectoryResult:
    return TrajectoryResult(
        objective=objective.name,
        setting=setting,
        seed=require_integer("seed", seed, minimum=0),
        best_so_far=list(state.best_so_far),
        final_dataset_size=len(state.rows),
        training_backend=TRAINING_BACKEND,
        fm_metadata=dict(state.fm_metadata),
        qubo_stats=state.qubo_stats,
        duplicate_replacements=state.duplicate_replacements,
        random_replacements=state.random_replacements,
        random_replacement_draws=state.random_replacement_draws,
        accepted_sa_candidates=state.accepted_sa_candidates,
        infeasible_sa_samples_skipped=state.infeasible_sa_samples_skipped,
        max_feasible_candidate_rank=state.max_feasible_candidate_rank,
        completed_iterations=state.completed_iterations,
    )


def run_single_trajectory(
    initial_rows: Sequence[Dict[str, Any]],
    objective: ObjectiveSpec,
    seed: int,
    config: Figure4ExperimentConfig,
    setting: Figure4Setting = "wo_cgfm",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> TrajectoryResult:
    """供 Figure 4 runner 和白盒测试使用的内部单 trajectory 编排函数。"""

    normalized_seed = _validate_seed_schedule([seed], config)[0]
    strategy = get_setting_strategy(setting)
    selected_setting = strategy.name
    trajectory_index = _trajectory_index(objective, selected_setting)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

    if resume and checkpoint is not None and checkpoint.exists():
        # resume 只接受 schema/config/setting/objective/seed 全部匹配的 checkpoint。
        LOGGER.info(
            "Loading checkpoint setting=%s objective=%s seed=%s path=%s",
            selected_setting,
            objective.name,
            seed,
            checkpoint,
        )
        loaded_checkpoint = load_checkpoint(
            checkpoint,
            objective=objective,
            seed=normalized_seed,
            config=config,
            setting=selected_setting,
        )
        state = _state_from_checkpoint(loaded_checkpoint, checkpoint)
        _validate_trajectory_state(
            state,
            objective=objective,
            setting=selected_setting,
            seed=normalized_seed,
            config=config,
            checkpoint=checkpoint,
        )
        if state.completed_iterations == config.iterations:
            LOGGER.info(
                "Skipping completed trajectory setting=%s objective=%s seed=%s completed=%s",
                selected_setting,
                objective.name,
                seed,
                state.completed_iterations,
            )
            return _result_from_state(
                objective=objective,
                setting=selected_setting,
                seed=normalized_seed,
                state=state,
            )
    else:
        state = _initial_state(initial_rows, config)

    seen_compositions = {_row_composition_key(row) for row in state.rows}
    best_value = _current_best_value(state, objective)
    LOGGER.info(
        "Starting trajectory setting=%s objective=%s seed=%s start_iteration=%s target_iterations=%s rows=%s",
        selected_setting,
        objective.name,
        seed,
        state.completed_iterations,
        config.iterations,
        len(state.rows),
    )

    for iteration in range(state.completed_iterations, config.iterations):
        replacement_rng = random.Random(
            derive_python_seed(
                FIGURE4_REPLACEMENT_NAMESPACE,
                normalized_seed,
                trajectory_index=trajectory_index,
                iteration=iteration,
            )
        )
        iteration_result = _fit_and_solve_iteration(
            rows=state.rows,
            objective=objective,
            strategy=strategy,
            config=config,
            seed=normalized_seed,
            iteration=iteration,
        )
        state.fm_metadata = iteration_result.fm_metadata
        state.qubo_stats = iteration_result.qubo_stats
        state.infeasible_sa_samples_skipped += iteration_result.infeasible_sa_samples_skipped
        state.max_feasible_candidate_rank = max(
            state.max_feasible_candidate_rank,
            iteration_result.feasible_candidate_rank,
        )

        decision = _decide_candidate(
            candidate_bits=iteration_result.candidate_bits,
            encoding=iteration_result.encoding,
            strategy=strategy,
            sample_id=len(state.rows),
            seed=normalized_seed,
            replacement_rng=replacement_rng,
            num_levels=config.num_levels,
            seen_compositions=seen_compositions,
        )
        _apply_candidate_decision(state, decision)
        seen_compositions.add(_row_composition_key(decision.row))
        # best_so_far 记录的是论文 Figure 4 曲线：每轮加入新样本后的历史最优。
        best_value = objective.update_best(best_value, float(decision.row[objective.name]))
        state.best_so_far.append(best_value)
        LOGGER.info(
            (
                "Completed iteration setting=%s objective=%s seed=%s iteration=%s/%s "
                "decision=%s best_so_far=%.12g rows=%s accepted_sa=%s duplicate_repl=%s "
                "feasible_rank=%s skipped_infeasible=%s"
            ),
            selected_setting,
            objective.name,
            seed,
            iteration + 1,
            config.iterations,
            decision.status,
            best_value,
            len(state.rows),
            state.accepted_sa_candidates,
            state.duplicate_replacements,
            iteration_result.feasible_candidate_rank,
            iteration_result.infeasible_sa_samples_skipped,
        )

        if checkpoint is not None:
            write_checkpoint(
                checkpoint,
                checkpoint_payload(
                    objective=objective,
                    setting=selected_setting,
                    seed=normalized_seed,
                    state=state,
                    config=config,
                ),
            )

    LOGGER.info(
        "Finished trajectory setting=%s objective=%s seed=%s completed=%s final_rows=%s",
        selected_setting,
        objective.name,
        seed,
        state.completed_iterations,
        len(state.rows),
    )
    return _result_from_state(
        objective=objective,
        setting=selected_setting,
        seed=normalized_seed,
        state=state,
    )


def aggregate_trajectory_results(results: Sequence[TrajectoryResult]) -> Dict[str, AggregatedTrajectory]:
    """按 objective + setting 聚合同一曲线的多 seed 结果。"""

    grouped: Dict[str, List[TrajectoryResult]] = {}
    for result in results:
        key = f"{result.objective}:{result.setting}"
        grouped.setdefault(key, []).append(result)

    aggregated: Dict[str, AggregatedTrajectory] = {}
    for key, trajectories in grouped.items():
        max_len = max(len(result.best_so_far) for result in trajectories)
        array = np.full((len(trajectories), max_len), np.nan, dtype=np.float64)
        for idx, result in enumerate(trajectories):
            array[idx, : len(result.best_so_far)] = result.best_so_far
        objective_name, setting_name = key.split(":", 1)
        aggregated[key] = AggregatedTrajectory(
            objective=objective_name,
            setting=setting_name,
            best_so_far_mean=np.nanmean(array, axis=0).tolist(),
            best_so_far_std=np.nanstd(array, axis=0).tolist(),
            best_so_far_min=np.nanmin(array, axis=0).tolist(),
            best_so_far_max=np.nanmax(array, axis=0).tolist(),
            num_trajectories=len(trajectories),
        )
    return aggregated


def run_figure4_experiment(
    seed_list: Sequence[int],
    config: Figure4ExperimentConfig,
    output_dir: str | Path,
    objectives: Sequence[ObjectiveSpec] = FIGURE4_OBJECTIVES,
    resume: bool = False,
    settings: Sequence[Figure4Setting] = FIGURE4_SETTINGS,
) -> Figure4Summary:
    """供 Figure 4 runner 和白盒测试使用的内部实验编排函数。"""

    selected_settings = validate_settings(settings)
    selected_seeds = _validate_seed_schedule(seed_list, config)

    output_layout = figure4_output_layout(output_dir)
    output_layout.root.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting Figure 4 experiment output=%s seeds=%s objectives=%s settings=%s",
        output_layout.root,
        selected_seeds,
        [objective.name for objective in objectives],
        list(selected_settings),
    )
    dataset_batch = generate_initial_dataset_batch(seed_list=selected_seeds, num_samples=config.num_samples)

    trajectory_results: List[TrajectoryResult] = []
    for seed in selected_seeds:
        initial_rows, _ = dataset_batch[int(seed)]
        for objective in objectives:
            for setting in selected_settings:
                trajectory_checkpoint_path = output_layout.checkpoint_path(setting, objective.name, int(seed))
                trajectory_results.append(
                    run_single_trajectory(
                        initial_rows=initial_rows,
                        objective=objective,
                        seed=int(seed),
                        config=config,
                        setting=setting,
                        checkpoint_path=trajectory_checkpoint_path,
                        resume=resume,
                    )
                )

    summary = Figure4Summary(
        schema_version=SUMMARY_SCHEMA_VERSION,
        seed_derivation=SEED_DERIVATION_SCHEME,
        training_backend=TRAINING_BACKEND,
        config=config.to_dict(),
        seed_list=selected_seeds,
        objectives=[objective.name for objective in objectives],
        settings=list(selected_settings),
        trajectories=trajectory_results,
        aggregated=aggregate_trajectory_results(trajectory_results),
    )
    write_json_atomic(output_layout.summary_path, summary.to_dict())
    LOGGER.info("Wrote Figure 4 summary path=%s", output_layout.summary_path)
    return summary

__all__ = [
    "AggregatedTrajectory",
    "Figure4Summary",
    "TrajectoryResult",
    "TrajectoryState",
    "aggregate_trajectory_results",
    "discretize_row_for_figure4",
    "discretize_rows_for_figure4",
]
