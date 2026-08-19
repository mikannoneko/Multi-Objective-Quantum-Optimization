"""Acceptance checks for the project's small-scale Figure 4/5 workflows."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alloy_dataset_generator import build_dataset_row
from experiment_runtime import SEED_DERIVATION_SCHEME, load_json_object, write_json_atomic
from figure4_experiment_config import (
    FIGURE4_OBJECTIVE_NAMES,
    FIGURE4_PRESET_NUM_SEEDS,
    FIGURE4_SETTINGS,
    preset_config as figure4_preset_config,
)
from figure4_outputs import SUMMARY_SCHEMA_VERSION as FIGURE4_SCHEMA_VERSION
from figure5_experiment_config import (
    FIGURE5_PRESET_NUM_SEEDS,
    preset_config as figure5_preset_config,
)
from figure5_outputs import SUMMARY_SCHEMA_VERSION as FIGURE5_SCHEMA_VERSION
from figure5_pareto import pareto_front
from figure5_scalarization import FIGURE5_SETTINGS


REPORT_SCHEMA_VERSION = 2


def _load_summary(path: str | Path) -> dict[str, Any]:
    return load_json_object(path, document_name="reproduction summary")


def _check(checks: dict[str, dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _matches_exact_string_set(values: Sequence[Any], expected: Sequence[str]) -> bool:
    return (
        len(values) == len(expected)
        and all(isinstance(value, str) for value in values)
        and set(values) == set(expected)
    )


def _finite_series(value: Any, expected_length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_length:
        return None
    try:
        series = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return series if all(math.isfinite(item) for item in series) else None


def _scale_detail(config: Mapping[str, Any], seeds: Sequence[Any]) -> dict[str, Any]:
    encoding = config.get("encoding") if isinstance(config.get("encoding"), Mapping) else {}
    return {
        "num_samples": config.get("num_samples"),
        "iterations": config.get("iterations"),
        "num_levels": encoding.get("num_levels"),
        "num_seeds": len(seeds),
    }


_FIGURE4_QUICK_SCALE = _scale_detail(
    figure4_preset_config("quick").to_dict(),
    range(FIGURE4_PRESET_NUM_SEEDS["quick"]),
)
_FIGURE5_QUICK_SCALE = _scale_detail(
    figure5_preset_config("quick").to_dict(),
    range(FIGURE5_PRESET_NUM_SEEDS["quick"]),
)


def _trajectory_accounting(
    trajectory: Mapping[str, Any],
    *,
    expected_iterations: int,
    expected_final_dataset_size: int,
) -> dict[str, Any]:
    accepted = _as_int(trajectory.get("accepted_sa_candidates"))
    duplicates = _as_int(trajectory.get("duplicate_replacements"))
    replacements = _as_int(trajectory.get("random_replacements"))
    replacement_draws = _as_int(trajectory.get("random_replacement_draws"))
    infeasible_skipped = _as_int(trajectory.get("infeasible_sa_samples_skipped"))
    max_rank = _as_int(trajectory.get("max_feasible_candidate_rank"))
    final_dataset_size = _as_int(trajectory.get("final_dataset_size"))
    passed = (
        accepted >= 0
        and duplicates >= 0
        and accepted + duplicates == expected_iterations
        and replacements == duplicates
        and replacement_draws >= replacements
        and infeasible_skipped >= 0
        and max_rank >= 1
        and final_dataset_size == expected_final_dataset_size
    )
    return {
        "passed": passed,
        "accepted_sa_candidates": accepted,
        "duplicate_replacements": duplicates,
        "random_replacements": replacements,
        "random_replacement_draws": replacement_draws,
        "infeasible_sa_samples_skipped": infeasible_skipped,
        "max_feasible_candidate_rank": max_rank,
        "final_dataset_size": final_dataset_size,
    }


def validate_figure4_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical Figure 4 quick workflow and expose trends as diagnostics."""

    checks: dict[str, dict[str, Any]] = {}
    _check(
        checks,
        "schema",
        summary.get("schema_version") == FIGURE4_SCHEMA_VERSION
        and summary.get("seed_derivation") == SEED_DERIVATION_SCHEME,
        {
            "expected": FIGURE4_SCHEMA_VERSION,
            "actual": summary.get("schema_version"),
            "seed_derivation": summary.get("seed_derivation"),
        },
    )
    raw_objectives = summary.get("objectives")
    objectives = raw_objectives if isinstance(raw_objectives, list) else []
    raw_settings = summary.get("settings")
    settings = raw_settings if isinstance(raw_settings, list) else []
    _check(
        checks,
        "comparison_axes",
        _matches_exact_string_set(objectives, FIGURE4_OBJECTIVE_NAMES)
        and _matches_exact_string_set(settings, FIGURE4_SETTINGS),
        {"objectives": objectives, "settings": settings},
    )

    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    seeds = summary.get("seed_list") if isinstance(summary.get("seed_list"), list) else []
    scale_detail = _scale_detail(config, seeds)
    _check(checks, "quick_scale", scale_detail == _FIGURE4_QUICK_SCALE, scale_detail)

    trajectories = summary.get("trajectories") if isinstance(summary.get("trajectories"), list) else []
    expected_iterations = _as_int(config.get("iterations"), 0)
    expected_trajectories = {
        (_as_int(seed), objective, setting)
        for seed in seeds
        for objective in FIGURE4_OBJECTIVE_NAMES
        for setting in FIGURE4_SETTINGS
    }
    actual_trajectories = {
        (_as_int(item.get("seed")), str(item.get("objective", "")), str(item.get("setting", "")))
        for item in trajectories
        if (
            isinstance(item, Mapping)
            and _as_int(item.get("completed_iterations")) == expected_iterations
            and isinstance(item.get("best_so_far"), list)
            and len(item["best_so_far"]) == expected_iterations
        )
    }
    complete = (
        bool(expected_trajectories)
        and len(trajectories) == len(expected_trajectories)
        and actual_trajectories == expected_trajectories
    )
    _check(
        checks,
        "complete_trajectories",
        complete,
        {
            "actual_count": len(actual_trajectories),
            "expected_count": len(expected_trajectories),
            "iterations": expected_iterations,
        },
    )

    aggregated = summary.get("aggregated") if isinstance(summary.get("aggregated"), Mapping) else {}
    aggregate_detail: dict[str, Any] = {}
    final_statistics: dict[str, tuple[float, float] | None] = {}
    aggregate_passed = True
    for objective in FIGURE4_OBJECTIVE_NAMES:
        for setting in FIGURE4_SETTINGS:
            key = f"{objective}:{setting}"
            entry = aggregated.get(key)
            valid = isinstance(entry, Mapping)
            curves: dict[str, list[float] | None] = {}
            if valid:
                for field in (
                    "best_so_far_mean",
                    "best_so_far_std",
                    "best_so_far_min",
                    "best_so_far_max",
                ):
                    curves[field] = _finite_series(entry.get(field), expected_iterations)
                valid = all(series is not None for series in curves.values())
                valid = valid and _as_int(entry.get("num_trajectories")) == len(seeds)
                std_curve = curves.get("best_so_far_std")
                valid = valid and std_curve is not None and all(value >= 0.0 for value in std_curve)
            aggregate_passed = aggregate_passed and valid
            aggregate_detail[key] = {"passed": valid, "num_points": expected_iterations}
            mean_curve = curves.get("best_so_far_mean")
            std_curve = curves.get("best_so_far_std")
            final_statistics[key] = (
                (mean_curve[-1], std_curve[-1])
                if valid and mean_curve is not None and std_curve is not None
                else None
            )
    _check(checks, "aggregate_curves", aggregate_passed, aggregate_detail)

    expected_final_dataset_size = _as_int(config.get("num_samples"), 0) + expected_iterations
    accounting = [
        _trajectory_accounting(
            item,
            expected_iterations=expected_iterations,
            expected_final_dataset_size=expected_final_dataset_size,
        )
        for item in trajectories
        if isinstance(item, Mapping)
    ]
    _check(
        checks,
        "candidate_accounting",
        len(accounting) == len(trajectories) and all(item["passed"] for item in accounting),
        accounting,
    )

    main_trends: dict[str, Any] = {}
    for objective, maximize in (("kappa", True), ("E", True), ("rho", False)):
        with_cgfm = final_statistics[f"{objective}:w_cgfm"]
        without_cgfm = final_statistics[f"{objective}:wo_cgfm"]
        observed = with_cgfm is not None and without_cgfm is not None
        if observed:
            observed = with_cgfm[0] > without_cgfm[0] if maximize else with_cgfm[0] < without_cgfm[0]
        main_trends[objective] = {
            "w_cgfm": with_cgfm,
            "wo_cgfm": without_cgfm,
            "paper_direction_observed": observed,
        }

    similarity: dict[str, Any] = {}
    for objective in ("delta_alpha", "delta_T"):
        with_cgfm = final_statistics[f"{objective}:w_cgfm"]
        without_cgfm = final_statistics[f"{objective}:wo_cgfm"]
        standardized_difference: float | None = None
        if with_cgfm is not None and without_cgfm is not None:
            spread = max(with_cgfm[1], without_cgfm[1], 1e-12)
            standardized_difference = abs(with_cgfm[0] - without_cgfm[0]) / spread
        similarity[objective] = {
            "w_cgfm": with_cgfm,
            "wo_cgfm": without_cgfm,
            "standardized_difference": standardized_difference,
        }
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "figure": 4,
        "scope": "quick_workflow",
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "diagnostics": {
            "cgfm_paper_direction": main_trends,
            "delta_metric_similarity": similarity,
        },
    }


def _composition_key(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("composition must contain four values")
    rounded = tuple(round(float(value), 10) for value in values)
    return rounded[0], rounded[1], rounded[2], rounded[3]


@lru_cache(maxsize=8)
def exact_figure5_front(num_levels: int) -> dict[tuple[float, float, float, float], tuple[float, float, float]]:
    """Enumerate the paper's direct grid and return its exact non-dominated front."""

    if int(num_levels) <= 1:
        raise ValueError("num_levels must be greater than 1")
    levels = int(num_levels)
    rows: list[dict[str, Any]] = []
    keys: list[tuple[float, float, float, float]] = []
    for first in range(levels + 1):
        for second in range(levels - first + 1):
            for third in range(levels - first - second + 1):
                fourth = levels - first - second - third
                composition = (first / levels, second / levels, third / levels, fourth / levels)
                rows.append(dict(build_dataset_row(len(rows), 0, composition)))
                keys.append(_composition_key(composition))

    oriented = np.asarray(
        [[float(row["kappa"]), float(row["E"]), -float(row["rho"])] for row in rows],
        dtype=np.float64,
    )
    non_dominated = np.ones(len(rows), dtype=bool)
    for index, candidate in enumerate(oriented):
        dominates = np.all(oriented >= candidate, axis=1) & np.any(oriented > candidate, axis=1)
        non_dominated[index] = not bool(np.any(dominates))
    return {
        keys[index]: tuple(float(value) for value in oriented[index])
        for index in np.flatnonzero(non_dominated)
    }


def _spacing_cv(oriented_points: Sequence[Sequence[float]], exact_values: np.ndarray) -> float | None:
    if len(oriented_points) < 2:
        return None
    points = np.asarray(oriented_points, dtype=np.float64)
    minimum = np.min(exact_values, axis=0)
    span = np.ptp(exact_values, axis=0)
    span = np.where(span > 0.0, span, 1.0)
    normalized = (points - minimum) / span
    distances = np.sqrt(np.sum((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    mean = float(np.mean(nearest))
    return 0.0 if mean <= 0.0 else float(np.std(nearest) / mean)


def _mean_std(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None, None
    array = np.asarray(finite, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def figure5_front_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    encoding = config.get("encoding") if isinstance(config.get("encoding"), Mapping) else {}
    num_levels = _as_int(encoding.get("num_levels"), 0)
    exact = exact_figure5_front(num_levels)
    exact_values = np.asarray(list(exact.values()), dtype=np.float64)
    solutions = summary.get("solutions") if isinstance(summary.get("solutions"), list) else []
    raw_seeds = summary.get("seed_list") if isinstance(summary.get("seed_list"), list) else []
    seeds = [_as_int(seed) for seed in raw_seeds]
    raw_settings = summary.get("settings") if isinstance(summary.get("settings"), list) else []
    selected_settings = [setting for setting in FIGURE5_SETTINGS if setting in raw_settings]

    by_setting_seed: list[dict[str, Any]] = []
    for seed in seeds:
        for setting in selected_settings:
            unique: dict[tuple[float, float, float, float], Mapping[str, Any]] = {}
            for point in solutions:
                if not (
                    isinstance(point, Mapping)
                    and point.get("setting") == setting
                    and _as_int(point.get("seed")) == seed
                ):
                    continue
                composition = point.get("composition")
                if isinstance(composition, list):
                    unique.setdefault(_composition_key(composition), point)
            exact_hits = [key for key in unique if key in exact]
            by_setting_seed.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "unique_proposals": len(unique),
                    "exact_front_hits": len(exact_hits),
                    "exact_front_coverage": len(exact_hits) / len(exact) if exact else 0.0,
                    "exact_front_precision": len(exact_hits) / len(unique) if unique else 0.0,
                    "spacing_cv": _spacing_cv([exact[key] for key in exact_hits], exact_values),
                }
            )

    setting_aggregates: dict[str, dict[str, Any]] = {}
    for setting in selected_settings:
        records = [record for record in by_setting_seed if record["setting"] == setting]
        coverage_mean, coverage_std = _mean_std(
            [record["exact_front_coverage"] for record in records]
        )
        precision_mean, precision_std = _mean_std(
            [record["exact_front_precision"] for record in records]
        )
        spacing_mean, spacing_std = _mean_std([record["spacing_cv"] for record in records])
        setting_aggregates[setting] = {
            "num_seeds": len(records),
            "exact_front_coverage_mean": coverage_mean,
            "exact_front_coverage_std": coverage_std,
            "exact_front_precision_mean": precision_mean,
            "exact_front_precision_std": precision_std,
            "spacing_cv_mean": spacing_mean,
            "spacing_cv_std": spacing_std,
        }
    return {
        "exact_front_size": len(exact),
        "by_setting_seed": by_setting_seed,
        "setting_aggregates": setting_aggregates,
    }


def validate_figure5_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical Figure 5 quick workflow and report front diagnostics."""

    checks: dict[str, dict[str, Any]] = {}
    _check(
        checks,
        "schema",
        summary.get("schema_version") == FIGURE5_SCHEMA_VERSION
        and summary.get("seed_derivation") == SEED_DERIVATION_SCHEME,
        {
            "expected": FIGURE5_SCHEMA_VERSION,
            "actual": summary.get("schema_version"),
            "seed_derivation": summary.get("seed_derivation"),
        },
    )
    raw_settings = summary.get("settings")
    settings = raw_settings if isinstance(raw_settings, list) else []
    _check(
        checks,
        "comparison_settings",
        _matches_exact_string_set(settings, FIGURE5_SETTINGS),
        settings,
    )

    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    seeds = summary.get("seed_list") if isinstance(summary.get("seed_list"), list) else []
    scale_detail = _scale_detail(config, seeds)
    _check(checks, "quick_scale", scale_detail == _FIGURE5_QUICK_SCALE, scale_detail)

    expected_iterations = _as_int(config.get("iterations"), 0)
    trajectories = summary.get("trajectories") if isinstance(summary.get("trajectories"), list) else []
    expected_trajectories = {(_as_int(seed), setting) for seed in seeds for setting in FIGURE5_SETTINGS}
    actual_trajectories = {
        (_as_int(item.get("seed")), str(item.get("setting", "")))
        for item in trajectories
        if (
            isinstance(item, Mapping)
            and _as_int(item.get("completed_iterations")) == expected_iterations
            and isinstance(item.get("iteration_records"), list)
            and len(item["iteration_records"]) == expected_iterations
        )
    }
    complete = (
        bool(expected_trajectories)
        and len(trajectories) == len(expected_trajectories)
        and actual_trajectories == expected_trajectories
    )
    _check(
        checks,
        "complete_trajectories",
        complete,
        {
            "actual_count": len(actual_trajectories),
            "expected_count": len(expected_trajectories),
            "iterations": expected_iterations,
        },
    )

    expected_final_dataset_size = _as_int(config.get("num_samples"), 0) + expected_iterations
    accounting = [
        _trajectory_accounting(
            item,
            expected_iterations=expected_iterations,
            expected_final_dataset_size=expected_final_dataset_size,
        )
        for item in trajectories
        if isinstance(item, Mapping)
    ]
    _check(
        checks,
        "candidate_accounting",
        len(accounting) == len(trajectories) and all(item["passed"] for item in accounting),
        accounting,
    )

    solutions = summary.get("solutions") if isinstance(summary.get("solutions"), list) else []
    expected_solution_ids = {
        (_as_int(seed), setting, iteration)
        for seed in seeds
        for setting in FIGURE5_SETTINGS
        for iteration in range(expected_iterations)
    }
    actual_solution_ids: set[tuple[int, str, int]] = set()
    valid_solution_records = True
    for point in solutions:
        if not isinstance(point, Mapping):
            valid_solution_records = False
            continue
        composition = point.get("composition")
        try:
            fractions = [float(value) for value in composition] if isinstance(composition, list) else []
            objectives = [float(point[name]) for name in ("kappa", "E", "rho")]
        except (KeyError, TypeError, ValueError, OverflowError):
            valid_solution_records = False
            continue
        valid_solution_records = valid_solution_records and (
            len(fractions) == 4
            and all(math.isfinite(value) and value >= 0.0 for value in fractions)
            and math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-8)
            and all(math.isfinite(value) for value in objectives)
        )
        actual_solution_ids.add(
            (
                _as_int(point.get("seed")),
                str(point.get("setting", "")),
                _as_int(point.get("iteration")),
            )
        )
    solution_records_passed = (
        valid_solution_records
        and len(solutions) == len(expected_solution_ids)
        and actual_solution_ids == expected_solution_ids
    )
    _check(
        checks,
        "solution_records",
        solution_records_passed,
        {"actual_count": len(solutions), "expected_count": len(expected_solution_ids)},
    )

    raw_fronts = summary.get("pareto_fronts") if isinstance(summary.get("pareto_fronts"), list) else []
    expected_front_keys = {(_as_int(seed), setting) for seed in seeds for setting in FIGURE5_SETTINGS}
    actual_front_keys: set[tuple[int, str]] = set()
    front_sizes: dict[str, int | None] = {}
    fronts_passed = isinstance(summary.get("pareto_fronts"), list)
    for front in raw_fronts:
        if not isinstance(front, Mapping):
            fronts_passed = False
            continue
        front_seed = _as_int(front.get("seed"))
        front_setting = str(front.get("setting", ""))
        front_key = (front_seed, front_setting)
        front_solutions = front.get("solutions")
        detail_key = f"{front_setting}:seed_{front_seed}"
        front_sizes[detail_key] = len(front_solutions) if isinstance(front_solutions, list) else None
        if front_key in actual_front_keys or front_key not in expected_front_keys or not isinstance(front_solutions, list):
            fronts_passed = False
            continue
        actual_front_keys.add(front_key)
        source_solutions = [
            point
            for point in solutions
            if isinstance(point, Mapping)
            and _as_int(point.get("seed")) == front_seed
            and point.get("setting") == front_setting
        ]
        try:
            expected_front = pareto_front(source_solutions)
        except (TypeError, ValueError):
            fronts_passed = False
            continue
        if not expected_front or list(front_solutions) != expected_front:
            fronts_passed = False
    fronts_passed = fronts_passed and actual_front_keys == expected_front_keys and len(raw_fronts) == len(
        expected_front_keys
    )
    _check(checks, "summary_pareto_front", fronts_passed, front_sizes)

    metrics: dict[str, Any] | None = None
    try:
        metrics = figure5_front_metrics(summary)
    except (TypeError, ValueError) as error:
        _check(checks, "front_metrics", False, str(error))
    if metrics is not None:
        _check(checks, "front_metrics", True, metrics)
        records = {
            (record["seed"], record["setting"]): record
            for record in metrics["by_setting_seed"]
        }
        by_seed: list[dict[str, Any]] = []
        for seed in [_as_int(value) for value in seeds]:
            ddts = records.get((seed, "w_ddts"))
            weighted = records.get((seed, "wo_ddts"))
            coverage_ratio: float | None = None
            spacing_ratio: float | None = None
            if ddts is not None and weighted is not None:
                denominator = float(weighted["exact_front_coverage"])
                if denominator != 0.0:
                    coverage_ratio = float(ddts["exact_front_coverage"]) / denominator
                ddts_spacing = ddts["spacing_cv"]
                weighted_spacing = weighted["spacing_cv"]
                if ddts_spacing is not None and weighted_spacing not in {None, 0.0}:
                    spacing_ratio = float(ddts_spacing) / float(weighted_spacing)
            by_seed.append(
                {
                    "seed": seed,
                    "coverage_ratio_w_ddts_over_wo_ddts": coverage_ratio,
                    "spacing_cv_ratio_w_ddts_over_wo_ddts": spacing_ratio,
                }
            )
        coverage_mean, coverage_std = _mean_std(
            [record["coverage_ratio_w_ddts_over_wo_ddts"] for record in by_seed]
        )
        spacing_mean, spacing_std = _mean_std(
            [record["spacing_cv_ratio_w_ddts_over_wo_ddts"] for record in by_seed]
        )
        comparison = {
            "by_seed": by_seed,
            "aggregate": {
                "num_seeds": len(by_seed),
                "coverage_ratio_mean": coverage_mean,
                "coverage_ratio_std": coverage_std,
                "spacing_cv_ratio_mean": spacing_mean,
                "spacing_cv_ratio_std": spacing_std,
            },
        }
    else:
        comparison = None

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "figure": 5,
        "scope": "quick_workflow",
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "metrics": metrics,
        "diagnostics": {"ddts_comparison": comparison},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Figure 4 or Figure 5 quick-workflow summary.")
    parser.add_argument("figure", choices=("figure4", "figure5"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = _load_summary(args.summary)
    report = (
        validate_figure4_summary(summary)
        if args.figure == "figure4"
        else validate_figure5_summary(summary)
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "exact_figure5_front",
    "figure5_front_metrics",
    "main",
    "validate_figure4_summary",
    "validate_figure5_summary",
]
