"""Figure 4 active-learning 主流程。

一条 trajectory 对应固定的 `setting + objective + seed`：从同一个初始数据集出发，
重复训练 FM、转 QUBO、从 SA reads 选择可行候选、处理重复 composition，并记录
该 objective 的 best-so-far 曲线。
"""

from __future__ import annotations

import importlib
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

from alloy_dataset_generator import build_dataset_row, generate_initial_dataset_batch, sample_single_objective_design
from figure4_experiment_config import OBJECTIVES, ExperimentConfig, ObjectiveSpec
from figure4_fm_torch import fit_torch_fm, fm_to_qubo
from figure4_outputs import (
    checkpoint_payload,
    figure4_output_layout,
    load_checkpoint,
    write_checkpoint,
    write_json_atomic,
)
from figure4_qubo_math import (
    IterationEncoding,
    QuboBuildResult,
    QuboStats,
    build_single_objective_qubo,
    prepare_discrete_composition,
    select_lowest_energy_feasible_sample,
    solve_qubo_with_sa,
    validate_candidate_composition,
)
from figure4_setting_strategies import (
    Figure4Setting,
    SettingStrategy,
    get_setting_strategy,
    validate_settings,
)
from experiment_runtime import validate_seed_list


TRAINING_REQUIRED_MODULES = ("torch", "optuna", "numpy", "scipy", "sklearn", "neal", "dimod")
SUMMARY_SCHEMA_VERSION = 3
TRAINING_BACKEND = "pytorch_fm_lbfgs"
MAX_RANDOM_REPLACEMENT_ATTEMPTS = 10_000
CandidateStatus = Literal["accepted", "duplicate_replacement"]
LOGGER = logging.getLogger(__name__)


def ensure_training_dependencies() -> None:
    missing = [name for name in TRAINING_REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing required dependencies: {', '.join(missing)}")


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
    return build_dataset_row(int(row["sample_id"]), int(row["seed"]), discrete_composition)


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


def _state_from_checkpoint(payload: Dict[str, Any]) -> TrajectoryState:
    raw_state = dict(payload["state"])
    raw_qubo_stats = raw_state.get("qubo_stats")
    qubo_stats = QuboStats(**raw_qubo_stats) if raw_qubo_stats is not None else None
    return TrajectoryState(
        rows=[dict(row) for row in raw_state.get("rows", [])],
        best_so_far=[float(value) for value in raw_state.get("best_so_far", [])],
        fm_metadata=dict(raw_state.get("fm_metadata", {})),
        qubo_stats=qubo_stats,
        duplicate_replacements=int(raw_state.get("duplicate_replacements", 0)),
        random_replacements=int(raw_state.get("random_replacements", 0)),
        random_replacement_draws=int(raw_state.get("random_replacement_draws", 0)),
        accepted_sa_candidates=int(raw_state.get("accepted_sa_candidates", 0)),
        infeasible_sa_samples_skipped=int(raw_state.get("infeasible_sa_samples_skipped", 0)),
        max_feasible_candidate_rank=int(raw_state.get("max_feasible_candidate_rank", 0)),
    )


def _advance_replacement_rng(rng: random.Random, replacement_draws: int) -> None:
    for _ in range(max(0, replacement_draws)):
        sample_single_objective_design(rng)


def _initial_state(initial_rows: Sequence[Dict[str, Any]], config: ExperimentConfig) -> TrajectoryState:
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
    config: ExperimentConfig,
    seed: int,
    iteration: int,
) -> TrainingIterationResult:
    """完成单轮“当前数据集 -> FM -> QUBO -> SA 候选 bit vector”。"""

    encoding = strategy.create_encoding(config.num_levels, seed + iteration)
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
        seed=seed + iteration,
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
        seed=seed + (iteration * 9973),
    )

    def is_feasible(candidate_bits: np.ndarray) -> bool:
        composition = strategy.decode_candidate(candidate_bits, encoding)
        return composition is not None and validate_candidate_composition(composition)

    selected = select_lowest_energy_feasible_sample(sampling, is_feasible)
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
        seed=int(seed),
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
    config: ExperimentConfig,
    setting: Figure4Setting = "wo_cgfm",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> TrajectoryResult:
    """运行或恢复一条固定 setting/objective/seed 的 Figure 4 trajectory。"""

    strategy = get_setting_strategy(setting)
    selected_setting = strategy.name
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
            seed=seed,
            config=config,
            setting=selected_setting,
        )
        state = _state_from_checkpoint(loaded_checkpoint)
        if state.completed_iterations >= config.iterations:
            LOGGER.info(
                "Skipping completed trajectory setting=%s objective=%s seed=%s completed=%s",
                selected_setting,
                objective.name,
                seed,
                state.completed_iterations,
            )
            return _result_from_state(objective=objective, setting=selected_setting, seed=seed, state=state)
    else:
        state = _initial_state(initial_rows, config)

    replacement_rng = random.Random(seed + 17)
    # 恢复运行时跳过已消耗的 replacement 抽样，保证断点续跑和一次性运行一致。
    _advance_replacement_rng(replacement_rng, state.random_replacement_draws)

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
        iteration_result = _fit_and_solve_iteration(
            rows=state.rows,
            objective=objective,
            strategy=strategy,
            config=config,
            seed=int(seed),
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
            seed=int(seed),
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
                    seed=int(seed),
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
    return _result_from_state(objective=objective, setting=selected_setting, seed=seed, state=state)


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
    config: ExperimentConfig,
    output_dir: str | Path,
    objectives: Sequence[ObjectiveSpec] = OBJECTIVES,
    resume: bool = False,
    settings: Sequence[Figure4Setting] = ("wo_cgfm",),
) -> Figure4Summary:
    """运行 Figure 4 的 seed/objective/setting 笛卡尔积，并写出 schema v3 summary。"""

    selected_settings = validate_settings(settings)
    selected_seeds = validate_seed_list(seed_list)
    if not objectives:
        raise ValueError("At least one objective is required")

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
    "ensure_training_dependencies",
    "run_figure4_experiment",
    "run_single_trajectory",
]
