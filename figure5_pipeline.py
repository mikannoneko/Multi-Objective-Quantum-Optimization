"""Figure 5 多目标 FM+QO active-learning 主流程。

一条 trajectory 对应固定的 `setting + seed`。`w_ddts` 每轮训练一个
DDTS 人工目标 FM；`wo_ddts` 每轮分别训练三个 objective FM，然后按
preference weights 合并 QUBO。两条流程都使用直接四相 one-hot 编码和
system penalty，不使用 CGFM。
"""

from __future__ import annotations

import importlib
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import numpy as np

from alloy_dataset_generator import (
    build_dataset_row,
    generate_initial_dataset_multi_objective_batch,
    sample_multi_objective_design,
)
from figure4_fm_torch import fit_torch_fm, fm_to_qubo
from figure4_qubo_math import (
    IterationEncoding,
    QuboBuildResult,
    QuboStats,
    build_single_objective_qubo,
    create_iteration_encoding,
    decode_candidate_bits_to_composition,
    encode_single_objective_rows,
    prepare_discrete_composition,
    simulated_annealing_qubo,
    validate_candidate_composition,
)
from figure5_experiment_config import Figure5ExperimentConfig
from figure5_outputs import (
    checkpoint_payload,
    figure5_output_layout,
    load_checkpoint,
    write_checkpoint,
    write_json_atomic,
)
from figure5_pareto import pareto_front
from figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    Figure5Setting,
    compute_ddts_targets,
    compute_individual_objective_targets,
    sample_preference_weights,
)


TRAINING_REQUIRED_MODULES = ("torch", "optuna", "numpy", "scipy", "sklearn", "neal", "dimod")
SUMMARY_SCHEMA_VERSION = 1
TRAINING_BACKEND = "pytorch_fm_lbfgs"
SUPPORTED_SETTINGS: tuple[Figure5Setting, ...] = ("w_ddts", "wo_ddts")
MAX_RANDOM_REPLACEMENT_ATTEMPTS = 10_000
PREFERENCE_SEED_OFFSET = 50_005
PREFERENCE_ITERATION_STRIDE = 97_409
CandidateStatus = Literal["accepted", "invalid_replacement", "duplicate_replacement"]
LOGGER = logging.getLogger(__name__)


def ensure_training_dependencies() -> None:
    missing = [name for name in TRAINING_REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing required dependencies: {', '.join(missing)}")


@dataclass(frozen=True)
class SolutionPoint:
    """A validated alloy design stored in trajectory and summary records."""

    sample_id: int
    composition: List[float]
    kappa: float
    E: float
    rho: float


@dataclass(frozen=True)
class IterationRecord:
    """One active-learning iteration, including proposed and actually added designs."""

    setting: str
    seed: int
    iteration: int
    weights: List[float]
    scalarization_method: str
    decision_status: CandidateStatus
    proposed_solution: SolutionPoint | None
    added_solution: SolutionPoint
    replacement_draws: int
    sa_energy: float


@dataclass
class Figure5TrajectoryState:
    """Serializable in-progress state for one setting/seed trajectory."""

    rows: List[Dict[str, Any]]
    iteration_records: List[IterationRecord] = field(default_factory=list)
    latest_weights: List[float] = field(default_factory=list)
    latest_fm_metadata: Dict[str, Any] = field(default_factory=dict)
    latest_scalarization_metadata: Dict[str, Any] = field(default_factory=dict)
    qubo_stats: QuboStats | None = None
    duplicate_replacements: int = 0
    invalid_replacements: int = 0
    random_replacements: int = 0
    random_replacement_draws: int = 0
    accepted_sa_candidates: int = 0

    @property
    def completed_iterations(self) -> int:
        return len(self.iteration_records)


@dataclass(frozen=True)
class Figure5TrajectoryResult:
    setting: str
    seed: int
    iteration_records: List[IterationRecord]
    final_dataset_size: int
    training_backend: str
    latest_weights: List[float]
    latest_fm_metadata: Dict[str, Any]
    latest_scalarization_metadata: Dict[str, Any]
    qubo_stats: QuboStats | None
    duplicate_replacements: int
    invalid_replacements: int
    random_replacements: int
    random_replacement_draws: int
    accepted_sa_candidates: int
    completed_iterations: int


@dataclass(frozen=True)
class CandidateDecision:
    status: CandidateStatus
    proposed_row: Dict[str, Any] | None
    added_row: Dict[str, Any]
    replacement_draws: int = 0


@dataclass(frozen=True)
class TrainingIterationResult:
    encoding: IterationEncoding
    candidate_bits: np.ndarray
    weights: tuple[float, ...]
    scalarization_method: str
    fm_metadata: Dict[str, Any]
    scalarization_metadata: Dict[str, Any]
    qubo_stats: QuboStats
    sa_energy: float


@dataclass(frozen=True)
class Figure5Summary:
    schema_version: int
    training_backend: str
    config: Dict[str, Any]
    seed_list: List[int]
    objectives: List[str]
    settings: List[str]
    trajectories: List[Figure5TrajectoryResult]
    solutions: List[Dict[str, Any]]
    pareto_front: Dict[str, List[Mapping[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_settings(settings: Sequence[str]) -> tuple[Figure5Setting, ...]:
    if not settings:
        raise ValueError("At least one setting is required")
    unknown = [setting for setting in settings if setting not in SUPPORTED_SETTINGS]
    if unknown:
        raise ValueError(f"Unsupported Figure 5 settings: {', '.join(unknown)}")
    return tuple(settings)  # type: ignore[return-value]


def normalized_composition_from_row(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
        dtype=np.float64,
    )


def discretize_row_for_figure5(row: Mapping[str, Any], num_levels: int) -> Dict[str, Any]:
    """Put a continuous initial design onto the direct Figure 5 fraction grid."""

    composition = prepare_discrete_composition(normalized_composition_from_row(row), num_levels)
    return dict(build_dataset_row(int(row["sample_id"]), int(row["seed"]), composition))


def discretize_rows_for_figure5(rows: Sequence[Mapping[str, Any]], num_levels: int) -> List[Dict[str, Any]]:
    return [discretize_row_for_figure5(row, num_levels) for row in rows]


def random_replacement_row(sample_id: int, seed: int, rng: random.Random, num_levels: int) -> Dict[str, Any]:
    continuous_design = sample_multi_objective_design(rng)
    composition = prepare_discrete_composition(continuous_design, num_levels)
    return dict(build_dataset_row(sample_id, seed, composition))


def preference_weights_for_iteration(seed: int, iteration: int) -> tuple[float, ...]:
    """Derive the same deterministic weight vector for both settings."""

    rng_seed = int(seed) + PREFERENCE_SEED_OFFSET + (int(iteration) * PREFERENCE_ITERATION_STRIDE)
    return sample_preference_weights(random.Random(rng_seed), num_objectives=len(FIGURE5_OBJECTIVES))


def _composition_key(composition: Sequence[float]) -> Tuple[float, float, float, float]:
    return tuple(round(float(value), 10) for value in composition)


def _row_composition_key(row: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return _composition_key(normalized_composition_from_row(row))


def _solution_point_from_row(row: Mapping[str, Any]) -> SolutionPoint:
    return SolutionPoint(
        sample_id=int(row["sample_id"]),
        composition=[float(value) for value in normalized_composition_from_row(row)],
        kappa=float(row["kappa"]),
        E=float(row["E"]),
        rho=float(row["rho"]),
    )


def _solution_point_from_payload(payload: Mapping[str, Any] | None) -> SolutionPoint | None:
    if payload is None:
        return None
    return SolutionPoint(
        sample_id=int(payload["sample_id"]),
        composition=[float(value) for value in payload["composition"]],
        kappa=float(payload["kappa"]),
        E=float(payload["E"]),
        rho=float(payload["rho"]),
    )


def _iteration_record_from_payload(payload: Mapping[str, Any]) -> IterationRecord:
    added_solution = _solution_point_from_payload(payload.get("added_solution"))
    if added_solution is None:
        raise ValueError("Checkpoint iteration record is missing added_solution")
    return IterationRecord(
        setting=str(payload["setting"]),
        seed=int(payload["seed"]),
        iteration=int(payload["iteration"]),
        weights=[float(value) for value in payload.get("weights", [])],
        scalarization_method=str(payload["scalarization_method"]),
        decision_status=payload["decision_status"],
        proposed_solution=_solution_point_from_payload(payload.get("proposed_solution")),
        added_solution=added_solution,
        replacement_draws=int(payload.get("replacement_draws", 0)),
        sa_energy=float(payload["sa_energy"]),
    )


def _state_from_checkpoint(payload: Mapping[str, Any]) -> Figure5TrajectoryState:
    raw_state = dict(payload["state"])
    raw_qubo_stats = raw_state.get("qubo_stats")
    qubo_stats = QuboStats(**raw_qubo_stats) if raw_qubo_stats is not None else None
    return Figure5TrajectoryState(
        rows=[dict(row) for row in raw_state.get("rows", [])],
        iteration_records=[
            _iteration_record_from_payload(record) for record in raw_state.get("iteration_records", [])
        ],
        latest_weights=[float(value) for value in raw_state.get("latest_weights", [])],
        latest_fm_metadata=dict(raw_state.get("latest_fm_metadata", {})),
        latest_scalarization_metadata=dict(raw_state.get("latest_scalarization_metadata", {})),
        qubo_stats=qubo_stats,
        duplicate_replacements=int(raw_state.get("duplicate_replacements", 0)),
        invalid_replacements=int(raw_state.get("invalid_replacements", 0)),
        random_replacements=int(raw_state.get("random_replacements", 0)),
        random_replacement_draws=int(raw_state.get("random_replacement_draws", 0)),
        accepted_sa_candidates=int(raw_state.get("accepted_sa_candidates", 0)),
    )


def _advance_replacement_rng(rng: random.Random, replacement_draws: int) -> None:
    for _ in range(max(0, int(replacement_draws))):
        sample_multi_objective_design(rng)


def _generate_unique_replacement_row(
    *,
    sample_id: int,
    seed: int,
    rng: random.Random,
    num_levels: int,
    seen_compositions: set[Tuple[float, float, float, float]],
    max_attempts: int = MAX_RANDOM_REPLACEMENT_ATTEMPTS,
) -> tuple[Dict[str, Any], int]:
    for attempt in range(1, max_attempts + 1):
        row = random_replacement_row(sample_id, seed, rng, num_levels)
        if _row_composition_key(row) not in seen_compositions:
            return row, attempt
    raise RuntimeError(f"Unable to generate a novel random replacement after {max_attempts} attempts")


def _fit_and_solve_iteration(
    *,
    rows: Sequence[Dict[str, Any]],
    setting: Figure5Setting,
    config: Figure5ExperimentConfig,
    seed: int,
    iteration: int,
) -> TrainingIterationResult:
    """Run one Figure 5 preprocessing, FM, QUBO and SA iteration."""

    encoding = create_iteration_encoding(config.num_levels, seed + iteration, num_blocks=4)
    features = encode_single_objective_rows(rows, encoding)
    weights = preference_weights_for_iteration(seed, iteration)
    fit_seed = seed + iteration

    if setting == "w_ddts":
        scalarization = compute_ddts_targets(rows, weights)
        model, metadata = fit_torch_fm(
            features,
            scalarization.targets,
            optuna_trials=config.optuna_trials,
            device=config.device,
            seed=fit_seed,
        )
        merged_q, merged_bias = fm_to_qubo(model)
        fm_metadata: Dict[str, Any] = {"artificial_target": dict(metadata)}
        scalarization_method = scalarization.method
        scalarization_metadata = dict(scalarization.metadata)
    elif setting == "wo_ddts":
        individual = compute_individual_objective_targets(rows)
        merged_q = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
        merged_bias = 0.0
        fm_metadata = {}
        for objective_index, objective in enumerate(FIGURE5_OBJECTIVES):
            model, metadata = fit_torch_fm(
                features,
                individual.targets[objective],
                optuna_trials=config.optuna_trials,
                device=config.device,
                seed=fit_seed,
            )
            objective_q, objective_bias = fm_to_qubo(model)
            merged_q += float(weights[objective_index]) * objective_q
            merged_bias += float(weights[objective_index]) * objective_bias
            fm_metadata[objective] = dict(metadata)
        scalarization_method = "weighted_sum"
        scalarization_metadata = {
            **individual.metadata,
            "merge_weights": {name: float(weight) for name, weight in zip(FIGURE5_OBJECTIVES, weights)},
            "merge_stage": "qubo",
        }
    else:
        raise ValueError(f"Unsupported Figure 5 setting: {setting}")

    qubo_result: QuboBuildResult = build_single_objective_qubo(
        merged_q,
        merged_bias,
        encoding,
        include_system_penalty=True,
    )
    candidate_bits, sa_energy = simulated_annealing_qubo(
        qubo_result.q,
        qubo_result.bias,
        runs=config.sa_runs,
        sweeps=config.sa_sweeps,
        seed=seed + (iteration * 9973),
    )
    return TrainingIterationResult(
        encoding=encoding,
        candidate_bits=candidate_bits,
        weights=weights,
        scalarization_method=scalarization_method,
        fm_metadata=fm_metadata,
        scalarization_metadata=scalarization_metadata,
        qubo_stats=qubo_result.stats,
        sa_energy=float(sa_energy),
    )


def _decide_candidate(
    *,
    candidate_bits: np.ndarray,
    encoding: IterationEncoding,
    sample_id: int,
    seed: int,
    replacement_rng: random.Random,
    num_levels: int,
    seen_compositions: set[Tuple[float, float, float, float]],
) -> CandidateDecision:
    composition = decode_candidate_bits_to_composition(candidate_bits, encoding)
    if composition is None or not validate_candidate_composition(composition):
        replacement, draws = _generate_unique_replacement_row(
            sample_id=sample_id,
            seed=seed,
            rng=replacement_rng,
            num_levels=num_levels,
            seen_compositions=seen_compositions,
        )
        return CandidateDecision("invalid_replacement", None, replacement, draws)

    proposed_row = dict(build_dataset_row(sample_id, seed, composition))
    if _composition_key(composition) in seen_compositions:
        replacement, draws = _generate_unique_replacement_row(
            sample_id=sample_id,
            seed=seed,
            rng=replacement_rng,
            num_levels=num_levels,
            seen_compositions=seen_compositions,
        )
        return CandidateDecision("duplicate_replacement", proposed_row, replacement, draws)
    return CandidateDecision("accepted", proposed_row, proposed_row)


def _apply_candidate_decision(state: Figure5TrajectoryState, decision: CandidateDecision) -> None:
    if decision.status == "accepted":
        state.accepted_sa_candidates += 1
    elif decision.status == "invalid_replacement":
        state.invalid_replacements += 1
        state.random_replacements += 1
        state.random_replacement_draws += decision.replacement_draws
    elif decision.status == "duplicate_replacement":
        state.duplicate_replacements += 1
        state.random_replacements += 1
        state.random_replacement_draws += decision.replacement_draws
    else:
        raise ValueError(f"Unsupported candidate status: {decision.status}")
    state.rows.append(decision.added_row)


def _result_from_state(setting: Figure5Setting, seed: int, state: Figure5TrajectoryState) -> Figure5TrajectoryResult:
    return Figure5TrajectoryResult(
        setting=setting,
        seed=int(seed),
        iteration_records=list(state.iteration_records),
        final_dataset_size=len(state.rows),
        training_backend=TRAINING_BACKEND,
        latest_weights=list(state.latest_weights),
        latest_fm_metadata=dict(state.latest_fm_metadata),
        latest_scalarization_metadata=dict(state.latest_scalarization_metadata),
        qubo_stats=state.qubo_stats,
        duplicate_replacements=state.duplicate_replacements,
        invalid_replacements=state.invalid_replacements,
        random_replacements=state.random_replacements,
        random_replacement_draws=state.random_replacement_draws,
        accepted_sa_candidates=state.accepted_sa_candidates,
        completed_iterations=state.completed_iterations,
    )


def run_single_trajectory(
    initial_rows: Sequence[Dict[str, Any]],
    setting: Figure5Setting,
    seed: int,
    config: Figure5ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> Figure5TrajectoryResult:
    """Run or resume one fixed setting/seed Figure 5 trajectory."""

    selected_setting = validate_settings([setting])[0]
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is not None and checkpoint.exists():
        loaded = load_checkpoint(checkpoint, setting=selected_setting, seed=seed, config=config)
        state = _state_from_checkpoint(loaded)
        if state.completed_iterations >= config.iterations:
            LOGGER.info(
                "Skipping completed Figure 5 trajectory setting=%s seed=%s completed=%s",
                selected_setting,
                seed,
                state.completed_iterations,
            )
            return _result_from_state(selected_setting, seed, state)
    else:
        state = Figure5TrajectoryState(rows=discretize_rows_for_figure5(initial_rows, config.num_levels))

    replacement_rng = random.Random(seed + 17)
    _advance_replacement_rng(replacement_rng, state.random_replacement_draws)
    seen_compositions = {_row_composition_key(row) for row in state.rows}
    LOGGER.info(
        "Starting Figure 5 trajectory setting=%s seed=%s start=%s target=%s rows=%s",
        selected_setting,
        seed,
        state.completed_iterations,
        config.iterations,
        len(state.rows),
    )

    for iteration in range(state.completed_iterations, config.iterations):
        training = _fit_and_solve_iteration(
            rows=state.rows,
            setting=selected_setting,
            config=config,
            seed=int(seed),
            iteration=iteration,
        )
        state.latest_weights = [float(value) for value in training.weights]
        state.latest_fm_metadata = dict(training.fm_metadata)
        state.latest_scalarization_metadata = dict(training.scalarization_metadata)
        state.qubo_stats = training.qubo_stats

        decision = _decide_candidate(
            candidate_bits=training.candidate_bits,
            encoding=training.encoding,
            sample_id=len(state.rows),
            seed=int(seed),
            replacement_rng=replacement_rng,
            num_levels=config.num_levels,
            seen_compositions=seen_compositions,
        )
        _apply_candidate_decision(state, decision)
        seen_compositions.add(_row_composition_key(decision.added_row))
        state.iteration_records.append(
            IterationRecord(
                setting=selected_setting,
                seed=int(seed),
                iteration=iteration,
                weights=list(state.latest_weights),
                scalarization_method=training.scalarization_method,
                decision_status=decision.status,
                proposed_solution=(
                    None if decision.proposed_row is None else _solution_point_from_row(decision.proposed_row)
                ),
                added_solution=_solution_point_from_row(decision.added_row),
                replacement_draws=decision.replacement_draws,
                sa_energy=training.sa_energy,
            )
        )

        LOGGER.info(
            "Completed Figure 5 iteration setting=%s seed=%s iteration=%s/%s decision=%s rows=%s",
            selected_setting,
            seed,
            iteration + 1,
            config.iterations,
            decision.status,
            len(state.rows),
        )
        if checkpoint is not None:
            write_checkpoint(
                checkpoint,
                checkpoint_payload(setting=selected_setting, seed=int(seed), state=state, config=config),
            )

    return _result_from_state(selected_setting, seed, state)


def _flatten_proposed_solutions(
    trajectories: Sequence[Figure5TrajectoryResult],
) -> List[Dict[str, Any]]:
    solutions: List[Dict[str, Any]] = []
    for trajectory in trajectories:
        for record in trajectory.iteration_records:
            if record.proposed_solution is None:
                continue
            solutions.append(
                {
                    "setting": record.setting,
                    "seed": record.seed,
                    "iteration": record.iteration,
                    "status": record.decision_status,
                    "weights": list(record.weights),
                    **asdict(record.proposed_solution),
                }
            )
    return solutions


def run_figure5_experiment(
    seed_list: Sequence[int],
    config: Figure5ExperimentConfig,
    output_dir: str | Path,
    resume: bool = False,
    settings: Sequence[Figure5Setting] = SUPPORTED_SETTINGS,
) -> Figure5Summary:
    """Run the setting/seed Cartesian product and write Figure 5 summary schema v1."""

    selected_settings = validate_settings(settings)
    if not seed_list:
        raise ValueError("seed_list must not be empty")

    output_layout = figure5_output_layout(output_dir)
    output_layout.root.mkdir(parents=True, exist_ok=True)
    dataset_batch = generate_initial_dataset_multi_objective_batch(
        seed_list=seed_list,
        num_samples=config.num_samples,
    )
    trajectories: List[Figure5TrajectoryResult] = []
    for seed in seed_list:
        initial_rows, _ = dataset_batch[int(seed)]
        for setting in selected_settings:
            trajectories.append(
                run_single_trajectory(
                    initial_rows=initial_rows,
                    setting=setting,
                    seed=int(seed),
                    config=config,
                    checkpoint_path=output_layout.checkpoint_path(setting, int(seed)),
                    resume=resume,
                )
            )

    solutions = _flatten_proposed_solutions(trajectories)
    fronts: Dict[str, List[Mapping[str, Any]]] = {}
    for setting in selected_settings:
        fronts[setting] = pareto_front([solution for solution in solutions if solution["setting"] == setting])

    summary = Figure5Summary(
        schema_version=SUMMARY_SCHEMA_VERSION,
        training_backend=TRAINING_BACKEND,
        config=config.to_dict(),
        seed_list=[int(seed) for seed in seed_list],
        objectives=list(FIGURE5_OBJECTIVES),
        settings=list(selected_settings),
        trajectories=trajectories,
        solutions=solutions,
        pareto_front=fronts,
    )
    write_json_atomic(output_layout.summary_path, summary.to_dict())
    LOGGER.info("Wrote Figure 5 summary path=%s", output_layout.summary_path)
    return summary


__all__ = [
    "CandidateDecision",
    "Figure5Summary",
    "Figure5TrajectoryResult",
    "Figure5TrajectoryState",
    "IterationRecord",
    "SolutionPoint",
    "SUPPORTED_SETTINGS",
    "discretize_row_for_figure5",
    "discretize_rows_for_figure5",
    "ensure_training_dependencies",
    "preference_weights_for_iteration",
    "run_figure5_experiment",
    "run_single_trajectory",
    "validate_settings",
]
