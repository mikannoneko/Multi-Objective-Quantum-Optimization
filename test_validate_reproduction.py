from __future__ import annotations

import contextlib
import io
import json
import unittest

from validate_reproduction import (
    exact_figure5_front,
    figure5_front_metrics,
    parse_args,
    validate_figure4_summary,
    validate_figure5_summary,
)


FIGURE4_OBJECTIVES = ("kappa", "E", "rho", "delta_alpha", "delta_T")
FIGURE4_SETTINGS = ("w_cgfm", "wo_cgfm")
FIGURE5_SETTINGS = ("w_ddts", "wo_ddts")


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
        for objective in FIGURE4_OBJECTIVES
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
        for objective in FIGURE4_OBJECTIVES
        for setting in FIGURE4_SETTINGS
    ]
    return {
        "schema_version": 3,
        "config": {"num_samples": 100, "iterations": iterations, "encoding": {"num_levels": 50}},
        "seed_list": seeds,
        "objectives": list(FIGURE4_OBJECTIVES),
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
        "schema_version": 2,
        "settings": list(FIGURE5_SETTINGS),
        "config": {"num_samples": 500, "iterations": iterations, "encoding": {"num_levels": 25}},
        "seed_list": [0],
        "trajectories": trajectories,
        "solutions": solutions,
        "pareto_front": {
            setting: [next(point for point in solutions if point["setting"] == setting)]
            for setting in FIGURE5_SETTINGS
        },
    }


class ReproductionValidationTests(unittest.TestCase):
    def test_figure4_accepts_quick_workflow_and_keeps_trends_diagnostic(self) -> None:
        summary = _figure4_summary()
        kappa_with_cgfm = summary["aggregated"]["kappa:w_cgfm"]  # type: ignore[index]
        kappa_with_cgfm["best_so_far_mean"] = [0.0] * 100

        report = validate_figure4_summary(summary)

        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["quick_scale"]["passed"])
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

        self.assertLess(
            metrics["settings"]["w_ddts"]["exact_front_coverage"],
            metrics["settings"]["wo_ddts"]["exact_front_coverage"],
        )
        self.assertTrue(report["passed"])
        self.assertLess(
            report["diagnostics"]["ddts_comparison"]["coverage_ratio_w_ddts_over_wo_ddts"],
            1.0,
        )
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

    def test_cli_exposes_no_paper_acceptance_profile(self) -> None:
        args = parse_args(["figure4", "--summary", "summary.json"])
        self.assertFalse(hasattr(args, "profile"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["figure4", "--summary", "summary.json", "--profile", "paper"])


if __name__ == "__main__":
    unittest.main()
