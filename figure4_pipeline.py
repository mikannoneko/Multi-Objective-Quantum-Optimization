from __future__ import annotations

import importlib
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

from alloy_dataset_generator import build_dataset_row, generate_initial_dataset_batch, sample_single_objective_design
from figure4_fm_torch import FMHyperParams, TorchFMRegressor, fit_torch_fm, fm_to_qubo
from figure4_qubo_math import (
    IterationEncoding,
    build_one_hot_penalty_matrix,
    build_single_objective_qubo,
    build_system_penalty_matrix,
    cgfm_angles_to_composition,
    cgfm_composition_to_angles,
    create_cgfm_iteration_encoding,
    create_iteration_encoding,
    decode_candidate_bits_to_cgfm_composition,
    decode_candidate_bits_to_composition,
    encode_cgfm_rows,
    encode_single_objective_rows,
    evaluate_qubo_energy,
    prepare_discrete_composition,
    simulated_annealing_qubo,
    validate_candidate_composition,
)


REQUIRED_MODULES = ("torch", "optuna", "numpy", "scipy", "sklearn", "matplotlib", "neal", "dimod")
Figure4Setting = Literal["wo_cgfm", "w_cgfm"]
SUPPORTED_SETTINGS: Tuple[Figure4Setting, ...] = ("wo_cgfm", "w_cgfm")


def ensure_runtime_dependencies() -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing required dependencies: {', '.join(missing)}")


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    maximize: bool

    def transform_for_training(self, values: np.ndarray) -> np.ndarray:
        return -values if self.maximize else values

    def initial_best(self) -> float:
        return -math.inf if self.maximize else math.inf

    def update_best(self, current_best: float, candidate: float) -> float:
        if self.maximize:
            return max(current_best, candidate)
        return min(current_best, candidate)


OBJECTIVES: Tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("kappa", maximize=True),
    ObjectiveSpec("E", maximize=True),
    ObjectiveSpec("rho", maximize=False),
    ObjectiveSpec("delta_alpha", maximize=False),
    ObjectiveSpec("delta_T", maximize=False),
)


@dataclass(frozen=True)
class Figure4Config:
    num_samples: int = 100
    iterations: int = 600
    num_levels: int = 50
    optuna_trials: int = 20
    sa_runs: int = 1000
    sa_sweeps: int = 3000
    device: str = "cpu"


@dataclass
class TrajectoryResult:
    objective: str
    setting: str
    seed: int
    history_best: List[float]
    final_dataset_size: int
    training_backend: str
    fm_hparams: Dict[str, Any]
    qubo_stats: Dict[str, Any] = field(default_factory=dict)
    duplicate_replacements: int = 0
    invalid_replacements: int = 0
    accepted_candidates: int = 0
    completed_iterations: int = 0


def normalized_composition_from_row(row: Dict[str, Any]) -> np.ndarray:
    return np.array([float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])], dtype=np.float64)


def zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        std = 1.0
    return ((values - mean) / std).astype(np.float32), mean, std


def prepare_paper_row(row: Dict[str, Any], num_levels: int) -> Dict[str, Any]:
    discrete_composition = prepare_discrete_composition(normalized_composition_from_row(row), num_levels)
    return build_dataset_row(int(row["sample_id"]), int(row["seed"]), discrete_composition)


def prepare_paper_rows(rows: Sequence[Dict[str, Any]], num_levels: int) -> List[Dict[str, Any]]:
    return [prepare_paper_row(row, num_levels) for row in rows]


def random_replacement_row(sample_id: int, seed: int, rng: random.Random, num_levels: int) -> Dict[str, Any]:
    continuous_design = sample_single_objective_design(rng)
    discrete_composition = prepare_discrete_composition(continuous_design, num_levels)
    return build_dataset_row(sample_id, seed, discrete_composition)


def _composition_key(composition: Sequence[float]) -> Tuple[float, float, float, float]:
    return tuple(round(float(value), 10) for value in composition)


def _json_ready_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: value for key, value in row.items()} for row in rows]


def _checkpoint_payload(
    *,
    rows: Sequence[Dict[str, Any]],
    result: TrajectoryResult,
    config: Figure4Config,
) -> Dict[str, Any]:
    payload = asdict(result)
    payload.update(
        {
            "schema_version": 1,
            "config": asdict(config),
            "rows": _json_ready_rows(rows),
        }
    )
    return payload


def _write_trajectory_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _load_trajectory_checkpoint(
    path: Path,
    objective: ObjectiveSpec,
    seed: int,
    config: Figure4Config,
    setting: Figure4Setting,
) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported checkpoint schema in {path}")
    if payload.get("objective") != objective.name or int(payload.get("seed")) != int(seed):
        raise ValueError(f"Checkpoint metadata does not match requested trajectory: {path}")
    if payload.get("setting") != setting:
        raise ValueError(f"Checkpoint setting is not {setting}: {path}")
    if payload.get("config") != asdict(config):
        raise ValueError(
            f"Checkpoint config does not match current config: {path}. "
            "Use a separate output directory for different run settings."
        )
    return payload


def _result_from_checkpoint(payload: Dict[str, Any]) -> TrajectoryResult:
    return TrajectoryResult(
        objective=str(payload["objective"]),
        setting=str(payload["setting"]),
        seed=int(payload["seed"]),
        history_best=[float(value) for value in payload.get("history_best", [])],
        final_dataset_size=int(payload.get("final_dataset_size", len(payload.get("rows", [])))),
        training_backend=str(payload.get("training_backend", "pytorch_fm_lbfgs")),
        fm_hparams=dict(payload.get("fm_hparams", {})),
        qubo_stats=dict(payload.get("qubo_stats", {})),
        duplicate_replacements=int(payload.get("duplicate_replacements", 0)),
        invalid_replacements=int(payload.get("invalid_replacements", 0)),
        accepted_candidates=int(payload.get("accepted_candidates", 0)),
        completed_iterations=int(payload.get("completed_iterations", len(payload.get("history_best", [])))),
    )


def _advance_replacement_rng(rng: random.Random, replacement_count: int) -> None:
    for _ in range(max(0, replacement_count)):
        sample_single_objective_design(rng)


def _validate_setting(setting: str) -> Figure4Setting:
    if setting not in SUPPORTED_SETTINGS:
        raise ValueError(f"Unsupported setting {setting!r}. Choices: {', '.join(SUPPORTED_SETTINGS)}")
    return setting  # type: ignore[return-value]


def _legacy_checkpoint_path(checkpoint_path: Path, setting: Figure4Setting) -> Path | None:
    if setting != "wo_cgfm":
        return None
    prefix = "wo_cgfm_"
    if not checkpoint_path.name.startswith(prefix):
        return None
    return checkpoint_path.with_name(checkpoint_path.name[len(prefix) :])


def migrate_legacy_checkpoint_if_needed(checkpoint_path: Path, setting: Figure4Setting, resume: bool) -> None:
    if not resume or checkpoint_path.exists():
        return
    legacy_path = _legacy_checkpoint_path(checkpoint_path, setting)
    if legacy_path is not None and legacy_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, checkpoint_path)


def _create_setting_encoding(setting: Figure4Setting, num_levels: int, seed: int) -> IterationEncoding:
    if setting == "w_cgfm":
        return create_cgfm_iteration_encoding(num_levels, seed)
    return create_iteration_encoding(num_levels, seed)


def _encode_rows_for_setting(
    rows: Sequence[Dict[str, Any]],
    encoding: IterationEncoding,
    setting: Figure4Setting,
) -> np.ndarray:
    if setting == "w_cgfm":
        return encode_cgfm_rows(rows, encoding)
    return encode_single_objective_rows(rows, encoding)


def _decode_candidate_for_setting(
    candidate_bits: np.ndarray,
    encoding: IterationEncoding,
    setting: Figure4Setting,
) -> np.ndarray | None:
    if setting == "w_cgfm":
        return decode_candidate_bits_to_cgfm_composition(candidate_bits, encoding)
    return decode_candidate_bits_to_composition(candidate_bits, encoding)


def run_single_trajectory(
    initial_rows: Sequence[Dict[str, Any]],
    objective: ObjectiveSpec,
    objective_index: int,
    seed: int,
    config: Figure4Config,
    setting: Figure4Setting = "wo_cgfm",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> TrajectoryResult:
    setting = _validate_setting(setting)
    del objective_index  # not used in the paper-aligned w/o CGFM path
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_payload: Dict[str, Any] | None = None
    if resume and checkpoint is not None and checkpoint.exists():
        checkpoint_payload = _load_trajectory_checkpoint(checkpoint, objective, seed, config, setting)
        checkpoint_result = _result_from_checkpoint(checkpoint_payload)
        if checkpoint_result.completed_iterations >= config.iterations:
            return checkpoint_result

    if checkpoint_payload is not None:
        rows = [dict(row) for row in checkpoint_payload.get("rows", [])]
        history_best = [float(value) for value in checkpoint_payload.get("history_best", [])]
        duplicate_replacements = int(checkpoint_payload.get("duplicate_replacements", 0))
        invalid_replacements = int(checkpoint_payload.get("invalid_replacements", 0))
        accepted_candidates = int(checkpoint_payload.get("accepted_candidates", 0))
        start_iteration = len(history_best)
        best_value = history_best[-1] if history_best else objective.initial_best()
        if not history_best:
            for row in rows:
                best_value = objective.update_best(best_value, float(row[objective.name]))
    else:
        rows = prepare_paper_rows(initial_rows, config.num_levels)
        history_best = []
        duplicate_replacements = 0
        invalid_replacements = 0
        accepted_candidates = 0
        start_iteration = 0
        best_value = objective.initial_best()
        for row in rows:
            best_value = objective.update_best(best_value, float(row[objective.name]))

    replacement_rng = random.Random(seed + 17)
    _advance_replacement_rng(replacement_rng, duplicate_replacements + invalid_replacements)

    seen_compositions = {_composition_key(normalized_composition_from_row(row)) for row in rows}
    last_qubo_stats: Dict[str, Any] = dict(checkpoint_payload.get("qubo_stats", {})) if checkpoint_payload else {}
    last_hparams: Dict[str, Any] = dict(checkpoint_payload.get("fm_hparams", {})) if checkpoint_payload else {}

    for iteration in range(start_iteration, config.iterations):
        encoding: IterationEncoding = _create_setting_encoding(setting, config.num_levels, seed + iteration)
        features = _encode_rows_for_setting(rows, encoding, setting)
        raw_targets = np.array([float(row[objective.name]) for row in rows], dtype=np.float64)
        transformed_targets = objective.transform_for_training(raw_targets)
        scaled_targets, _, _ = zscore(transformed_targets)

        model, fit_metadata = fit_torch_fm(
            features,
            scaled_targets,
            optuna_trials=config.optuna_trials,
            device=config.device,
            seed=seed + iteration,
        )
        fm_q, fm_bias = fm_to_qubo(model)
        qubo_q, qubo_bias, qubo_stats = build_single_objective_qubo(
            fm_q,
            fm_bias,
            encoding,
            include_system_penalty=(setting == "wo_cgfm"),
        )
        last_qubo_stats = dict(qubo_stats)
        last_hparams = dict(fit_metadata)

        candidate_bits, _ = simulated_annealing_qubo(
            qubo_q,
            qubo_bias,
            runs=config.sa_runs,
            sweeps=config.sa_sweeps,
            seed=seed + (iteration * 9973),
        )
        candidate_composition = _decode_candidate_for_setting(candidate_bits, encoding, setting)
        invalid_candidate = candidate_composition is None or not validate_candidate_composition(candidate_composition)
        duplicate_candidate = False
        if not invalid_candidate:
            duplicate_candidate = _composition_key(candidate_composition) in seen_compositions

        if invalid_candidate or duplicate_candidate:
            if invalid_candidate:
                invalid_replacements += 1
            else:
                duplicate_replacements += 1
            new_row = random_replacement_row(len(rows), seed, replacement_rng, config.num_levels)
        else:
            new_row = build_dataset_row(len(rows), seed, candidate_composition)
            accepted_candidates += 1

        rows.append(new_row)
        seen_compositions.add(_composition_key(normalized_composition_from_row(new_row)))
        best_value = objective.update_best(best_value, float(new_row[objective.name]))
        history_best.append(best_value)

        if checkpoint is not None:
            partial_result = TrajectoryResult(
                objective=objective.name,
                setting=setting,
                seed=seed,
                history_best=history_best,
                final_dataset_size=len(rows),
                training_backend="pytorch_fm_lbfgs",
                fm_hparams=last_hparams,
                qubo_stats=last_qubo_stats,
                duplicate_replacements=duplicate_replacements,
                invalid_replacements=invalid_replacements,
                accepted_candidates=accepted_candidates,
                completed_iterations=len(history_best),
            )
            _write_trajectory_checkpoint(
                checkpoint,
                _checkpoint_payload(rows=rows, result=partial_result, config=config),
            )

    return TrajectoryResult(
        objective=objective.name,
        setting=setting,
        seed=seed,
        history_best=history_best,
        final_dataset_size=len(rows),
        training_backend="pytorch_fm_lbfgs",
        fm_hparams=last_hparams,
        qubo_stats=last_qubo_stats,
        duplicate_replacements=duplicate_replacements,
        invalid_replacements=invalid_replacements,
        accepted_candidates=accepted_candidates,
        completed_iterations=len(history_best),
    )


def aggregate_trajectory_results(results: Sequence[TrajectoryResult]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[TrajectoryResult]] = {}
    for result in results:
        key = f"{result.objective}:{result.setting}"
        grouped.setdefault(key, []).append(result)

    aggregated: Dict[str, Dict[str, Any]] = {}
    for key, trajectories in grouped.items():
        max_len = max(len(result.history_best) for result in trajectories)
        array = np.full((len(trajectories), max_len), np.nan, dtype=np.float64)
        for idx, result in enumerate(trajectories):
            array[idx, : len(result.history_best)] = result.history_best
        objective_name, setting_name = key.split(":", 1)
        aggregated[key] = {
            "objective": objective_name,
            "setting": setting_name,
            "mean_curve": np.nanmean(array, axis=0).tolist(),
            "std_curve": np.nanstd(array, axis=0).tolist(),
            "min_curve": np.nanmin(array, axis=0).tolist(),
            "max_curve": np.nanmax(array, axis=0).tolist(),
            "num_trajectories": len(trajectories),
        }
    return aggregated


def run_figure4_experiment(
    seed_list: Sequence[int],
    config: Figure4Config,
    output_dir: str | Path,
    objectives: Sequence[ObjectiveSpec] = OBJECTIVES,
    resume: bool = False,
    settings: Sequence[Figure4Setting] = ("wo_cgfm",),
) -> Dict[str, Any]:
    selected_settings = tuple(_validate_setting(setting) for setting in settings)
    if not selected_settings:
        raise ValueError("At least one setting is required.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_path / "trajectories"
    dataset_batch = generate_initial_dataset_batch(seed_list=seed_list, num_samples=config.num_samples)

    trajectory_results: List[TrajectoryResult] = []
    for seed in seed_list:
        initial_rows, _ = dataset_batch[int(seed)]
        for objective_index, objective in enumerate(objectives):
            for setting in selected_settings:
                checkpoint_path = trajectory_path / f"{setting}_{objective.name}_seed_{int(seed)}.json"
                migrate_legacy_checkpoint_if_needed(checkpoint_path, setting, resume)
                trajectory_results.append(
                    run_single_trajectory(
                        initial_rows=initial_rows,
                        objective=objective,
                        objective_index=objective_index,
                        seed=int(seed),
                        config=config,
                        setting=setting,
                        checkpoint_path=checkpoint_path,
                        resume=resume,
                    )
                )

    aggregated = aggregate_trajectory_results(trajectory_results)
    summary = {
        "training_backend": "pytorch_fm_lbfgs",
        "config": asdict(config),
        "seed_list": [int(seed) for seed in seed_list],
        "objectives": [objective.name for objective in objectives],
        "available_settings": list(selected_settings),
        "trajectories": [asdict(result) for result in trajectory_results],
        "aggregated": aggregated,
    }
    summary_path = output_path / "figure4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_summary(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "Figure4Config",
    "Figure4Setting",
    "FMHyperParams",
    "OBJECTIVES",
    "ObjectiveSpec",
    "TorchFMRegressor",
    "TrajectoryResult",
    "aggregate_trajectory_results",
    "build_one_hot_penalty_matrix",
    "build_system_penalty_matrix",
    "cgfm_angles_to_composition",
    "cgfm_composition_to_angles",
    "create_cgfm_iteration_encoding",
    "create_iteration_encoding",
    "decode_candidate_bits_to_cgfm_composition",
    "decode_candidate_bits_to_composition",
    "encode_cgfm_rows",
    "encode_single_objective_rows",
    "ensure_runtime_dependencies",
    "evaluate_qubo_energy",
    "fm_to_qubo",
    "load_summary",
    "prepare_discrete_composition",
    "run_figure4_experiment",
    "run_single_trajectory",
    "simulated_annealing_qubo",
    "SUPPORTED_SETTINGS",
    "validate_candidate_composition",
]
