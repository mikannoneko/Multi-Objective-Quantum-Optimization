from __future__ import annotations

import contextlib
import io
import json
import unittest

from experiment_runtime import SEED_DERIVATION_SCHEME
from figure4_experiment_config import FIGURE4_OBJECTIVE_NAMES, FIGURE4_SETTINGS
from figure5_pareto import pareto_front
from figure5_scalarization import FIGURE5_SETTINGS
from validate_reproduction import (
    exact_figure5_front,
    figure5_front_metrics,
    parse_args,
    validate_figure4_summary,
    validate_figure5_summary,
)


def _figure4_summary() -> dict[str, object]:
    iterations = 100
    seeds = [0, 1, 2]
    final_values = {
        "kappa": {"w_cgfm": (2.0, 0.2), "wo_cgfm": (1.0, 0.2)},
        "E": {"w_cgfm": (3.0, 0.2), "wo_cgfm": (2.0, 0.2)},
        "rho": {"w_cgfm": (1.0, 0.2), "wo_cgfm": (2.0, 0.2)},
        "delta_alpha": {"w_cgfm": (1.1, 0.2), "wo_cgfm": (1.0, 0.2)},
        "delta_T": {"w_cgfm": (0.9, 0.2), "wo_cgfm": (1.0, 0.2)},
    }
    aggregated = {
        f"{objective}:{setting}": {
            "best_so_far_mean": [final_values[objective][setting][0]] * iterations,
            "best_so_far_std": [final_values[objective][setting][1]] * iterations,
            "best_so_far_min": [final_values[objective][setting][0] - 0.1] * iterations,
            "best_so_far_max": [final_values[objective][setting][0] + 0.1] * iterations,
            "num_trajectories": len(seeds),
        }
        for objective in FIGURE4_OBJECTIVE_NAMES
        for setting in FIGURE4_SETTINGS
    }
    trajectories = [
        {
            "seed": seed,
            "objective": objective,
            "setting": setting,
            "completed_iterations": iterations,
            "best_so_far": [0.0] * iterations,
            "final_dataset_size": 200,
            "accepted_sa_candidates": iterations,
            "duplicate_replacements": 0,
            "random_replacements": 0,
            "random_replacement_draws": 0,
            "infeasible_sa_samples_skipped": 0,
            "max_feasible_candidate_rank": 1,
        }
        for seed in seeds
        for objective in FIGURE4_OBJECTIVE_NAMES
        for setting in FIGURE4_SETTINGS
    ]
    return {
        "schema_version": 4,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "config": {"num_samples": 100, "iterations": iterations, "encoding": {"num_levels": 50}},
        "seed_list": seeds,
        "objectives": list(FIGURE4_OBJECTIVE_NAMES),
        "settings": list(FIGURE4_SETTINGS),
        "trajectories": trajectories,
        "aggregated": aggregated,
    }


def _figure5_summary() -> dict[str, object]:
    iterations = 150
    exact = exact_figure5_front(25)
    exact_keys = list(exact)
    selected_keys = {
        "w_ddts": exact_keys[:5],
        "wo_ddts": exact_keys[:20],
    }
    solutions: list[dict[str, object]] = []
    for setting in FIGURE5_SETTINGS:
        keys = selected_keys[setting]
        for iteration in range(iterations):
            composition = keys[iteration % len(keys)]
            kappa, e_value, negative_rho = exact[composition]
            solutions.append(
                {
                    "setting": setting,
                    "seed": 0,
                    "iteration": iteration,
                    "composition": list(composition),
                    "kappa": kappa,
                    "E": e_value,
                    "rho": -negative_rho,
                }
            )
    trajectories = [
        {
            "seed": 0,
            "setting": setting,
            "completed_iterations": iterations,
            "iteration_records": [{} for _ in range(iterations)],
            "final_dataset_size": 650,
            "accepted_sa_candidates": iterations,
            "duplicate_replacements": 0,
            "random_replacements": 0,
            "random_replacement_draws": 0,
            "infeasible_sa_samples_skipped": 0,
            "max_feasible_candidate_rank": 1,
        }
        for setting in FIGURE5_SETTINGS
    ]
    return {
        "schema_version": 4,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "settings": list(FIGURE5_SETTINGS),
        "config": {"num_samples": 500, "iterations": iterations, "encoding": {"num_levels": 25}},
        "seed_list": [0],
        "trajectories": trajectories,
        "solutions": solutions,
        "pareto_fronts": [
            {
                "setting": setting,
                "seed": 0,
                "solutions": pareto_front([point for point in solutions if point["setting"] == setting]),
            }
            for setting in FIGURE5_SETTINGS
        ],
    }


class ReproductionValidationTests(unittest.TestCase):
    def test_figure4_accepts_quick_workflow_and_keeps_trends_diagnostic(self) -> None:
        summary = _figure4_summary()
        kappa_with_cgfm = summary["aggregated"]["kappa:w_cgfm"]  # type: ignore[index]
        kappa_with_cgfm["best_so_far_mean"] = [0.0] * 100

        report = validate_figure4_summary(summary)

        self.assertEqual(report["report_schema_version"], 2)
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["quick_scale"]["passed"])
        self.assertEqual(
            list(report["checks"]["aggregate_curves"]["detail"])[:2],
            ["kappa:wo_cgfm", "kappa:w_cgfm"],
        )
        self.assertFalse(
            report["diagnostics"]["cgfm_paper_direction"]["kappa"]["paper_direction_observed"]
        )

    def test_figure4_rejects_non_quick_scale(self) -> None:
        summary = _figure4_summary()
        summary["config"]["iterations"] = 600  # type: ignore[index]

        report = validate_figure4_summary(summary)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["quick_scale"]["passed"])

    def test_figure4_requires_every_objective_setting_trajectory(self) -> None:
        summary = _figure4_summary()
        summary["trajectories"] = summary["trajectories"][:-1]  # type: ignore[index]

        report = validate_figure4_summary(summary)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["complete_trajectories"]["passed"])

    def test_non_finite_figure4_statistics_fail_without_invalid_json(self) -> None:
        summary = _figure4_summary()
        kappa_with_cgfm = summary["aggregated"]["kappa:w_cgfm"]  # type: ignore[index]
        kappa_with_cgfm["best_so_far_mean"] = [float("nan")] * 100

        report = validate_figure4_summary(summary)

        self.assertFalse(report["passed"])
        json.dumps(report, allow_nan=False)

    def test_figure5_accepts_quick_workflow_without_scientific_thresholds(self) -> None:
        summary = _figure5_summary()
        metrics = figure5_front_metrics(summary)
        report = validate_figure5_summary(summary)

        self.assertEqual(report["report_schema_version"], 2)
        self.assertLess(
            metrics["setting_aggregates"]["w_ddts"]["exact_front_coverage_mean"],
            metrics["setting_aggregates"]["wo_ddts"]["exact_front_coverage_mean"],
        )
        self.assertTrue(report["passed"])
        self.assertLess(
            report["diagnostics"]["ddts_comparison"]["aggregate"]["coverage_ratio_mean"],
            1.0,
        )
        json.dumps(report, allow_nan=False)

    def test_figure5_metrics_are_isolated_by_seed_and_zero_denominators_are_null(self) -> None:
        exact = exact_figure5_front(2)
        first, second = list(exact)[:2]

        def point(setting: str, seed: int, iteration: int, composition: tuple[float, ...]) -> dict[str, object]:
            kappa, e_value, negative_rho = exact[composition]
            return {
                "setting": setting,
                "seed": seed,
                "iteration": iteration,
                "composition": list(composition),
                "kappa": kappa,
                "E": e_value,
                "rho": -negative_rho,
            }

        summary = {
            "schema_version": 4,
            "seed_derivation": SEED_DERIVATION_SCHEME,
            "settings": list(FIGURE5_SETTINGS),
            "config": {"num_samples": 500, "iterations": 150, "encoding": {"num_levels": 2}},
            "seed_list": [0, 1],
            "trajectories": [],
            "solutions": [
                point("w_ddts", 0, 0, first),
                point("w_ddts", 0, 1, first),
                point("w_ddts", 1, 0, second),
                point("wo_ddts", 1, 0, first),
            ],
            "pareto_fronts": [],
        }
        metrics = figure5_front_metrics(summary)
        records = metrics["by_setting_seed"]
        self.assertEqual(
            [(record["seed"], record["setting"]) for record in records],
            [(0, "w_ddts"), (0, "wo_ddts"), (1, "w_ddts"), (1, "wo_ddts")],
        )
        self.assertEqual(records[0]["unique_proposals"], 1)
        self.assertEqual(records[1]["unique_proposals"], 0)
        self.assertEqual(records[2]["unique_proposals"], 1)
        self.assertIsNone(metrics["setting_aggregates"]["w_ddts"]["spacing_cv_mean"])

        report = validate_figure5_summary(summary)
        comparison = report["diagnostics"]["ddts_comparison"]
        self.assertIsNone(comparison["by_seed"][0]["coverage_ratio_w_ddts_over_wo_ddts"])
        self.assertIsNone(comparison["aggregate"]["spacing_cv_ratio_mean"])
        json.dumps(report, allow_nan=False)

    def test_figure5_rejects_missing_comparison_setting(self) -> None:
        summary = _figure5_summary()
        summary["settings"] = ["w_ddts"]
        summary["solutions"] = [
            point for point in summary["solutions"] if point["setting"] == "w_ddts"  # type: ignore[index]
        ]

        report = validate_figure5_summary(summary)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["comparison_settings"]["passed"])
        self.assertFalse(report["checks"]["solution_records"]["passed"])
        json.dumps(report, allow_nan=False)

    def test_figure5_rejects_non_quick_scale(self) -> None:
        summary = _figure5_summary()
        summary["config"]["iterations"] = 1000  # type: ignore[index]

        report = validate_figure5_summary(summary)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["quick_scale"]["passed"])

    def test_figure5_rejects_schema_v3_and_inconsistent_stored_front(self) -> None:
        old_summary = _figure5_summary()
        old_summary["schema_version"] = 3
        old_report = validate_figure5_summary(old_summary)
        self.assertFalse(old_report["checks"]["schema"]["passed"])

        stale_summary = _figure5_summary()
        stale_summary["pareto_fronts"][0]["solutions"] = []  # type: ignore[index]
        stale_report = validate_figure5_summary(stale_summary)
        self.assertFalse(stale_report["checks"]["summary_pareto_front"]["passed"])

    def test_cli_exposes_no_paper_acceptance_profile(self) -> None:
        args = parse_args(["figure4", "--summary", "summary.json"])
        self.assertFalse(hasattr(args, "profile"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["figure4", "--summary", "summary.json", "--profile", "paper"])


if __name__ == "__main__":
    unittest.main()
