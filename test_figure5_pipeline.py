from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import figure5_pipeline as pipeline
import figure5_runner
from alloy_dataset_generator import generate_initial_dataset_multi_objective
from figure4_experiment_config import EncodingConfig, FMConfig, SAConfig
from figure4_qubo_math import QuboStats, build_single_objective_qubo, create_iteration_encoding, encode_discrete_composition
from figure5_experiment_config import Figure5ExperimentConfig, preset_config, resolve_experiment_config
from figure5_outputs import CHECKPOINT_SCHEMA_VERSION, checkpoint_payload, figure5_output_layout, load_checkpoint, write_checkpoint
from figure5_pipeline import (
    Figure5TrajectoryResult,
    IterationRecord,
    SolutionPoint,
    TrainingIterationResult,
    preference_weights_for_iteration,
    run_figure5_experiment,
    run_single_trajectory,
)


WORKSPACE_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_test"
WORKSPACE_TMP_ROOT.mkdir(exist_ok=True)


def _test_config(iterations: int = 2) -> Figure5ExperimentConfig:
    return Figure5ExperimentConfig(
        num_samples=10,
        iterations=iterations,
        encoding=EncodingConfig(num_levels=8),
        fm=FMConfig(optuna_trials=0, device="cpu"),
        sa=SAConfig(runs=2, sweeps=6),
    )


def _qubo_stats(num_variables: int = 32) -> QuboStats:
    return QuboStats(
        fm_scale=1.0,
        system_scale=1.0,
        one_hot_scale=1.0,
        system_penalty_weight=650.0,
        one_hot_penalty_weight=1.0,
        num_variables=num_variables,
        max_abs=1.0,
    )


def _training_result(config: Figure5ExperimentConfig, seed: int, iteration: int) -> TrainingIterationResult:
    encoding = create_iteration_encoding(config.num_levels, seed + iteration, num_blocks=4)
    return TrainingIterationResult(
        encoding=encoding,
        candidate_bits=np.zeros(4 * config.num_levels, dtype=np.float64),
        weights=preference_weights_for_iteration(seed, iteration),
        scalarization_method="ddts",
        fm_metadata={"mock": True},
        scalarization_metadata={"mock": True},
        qubo_stats=_qubo_stats(4 * config.num_levels),
        sa_energy=0.0,
    )


class Figure5PipelineTests(unittest.TestCase):
    def test_config_presets_overrides_and_runner_default(self) -> None:
        paper = preset_config("paper", device="cpu")
        quick = preset_config("quick150", device="cuda")
        test = preset_config("test", device="cpu")

        self.assertEqual((paper.num_samples, paper.iterations, paper.num_levels), (500, 1000, 25))
        self.assertEqual((quick.num_samples, quick.iterations, quick.num_levels), (500, 150, 25))
        self.assertEqual((quick.optuna_trials, quick.sa_runs, quick.sa_sweeps), (3, 100, 500))
        self.assertEqual((test.num_samples, test.iterations, test.num_levels), (10, 2, 8))

        overridden = resolve_experiment_config(
            preset="test",
            device="cpu",
            iterations=4,
            num_levels=9,
            sa_runs=7,
        )
        self.assertEqual((overridden.iterations, overridden.num_levels, overridden.sa_runs), (4, 9, 7))

        args = figure5_runner.parse_args(["--output-dir", "figure5_default", "--settings", "w_ddts"])
        self.assertEqual(args.preset, "quick150")
        self.assertEqual(figure5_runner.PRESET_NUM_SEEDS["quick150"], 3)

    def test_preference_weights_are_shared_and_deterministic(self) -> None:
        weights_a = preference_weights_for_iteration(seed=4, iteration=7)
        weights_b = preference_weights_for_iteration(seed=4, iteration=7)
        weights_c = preference_weights_for_iteration(seed=4, iteration=8)

        self.assertEqual(weights_a, weights_b)
        self.assertNotEqual(weights_a, weights_c)
        self.assertAlmostEqual(sum(weights_a), 1.0, places=12)

    def test_training_uses_one_fm_for_ddts_and_three_for_weighted_sum(self) -> None:
        config = _test_config(iterations=1)
        rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=2)
        rows = pipeline.discretize_rows_for_figure5(rows, config.num_levels)
        size = 4 * config.num_levels

        def fake_sa(q: np.ndarray, bias: float, **_: object) -> tuple[np.ndarray, float]:
            return np.zeros(q.shape[0], dtype=np.float64), float(bias)

        with (
            mock.patch("figure5_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})) as fit_mock,
            mock.patch("figure5_pipeline.fm_to_qubo", return_value=(np.eye(size), 1.0)),
            mock.patch("figure5_pipeline.simulated_annealing_qubo", side_effect=fake_sa),
        ):
            ddts = pipeline._fit_and_solve_iteration(
                rows=rows,
                setting="w_ddts",
                config=config,
                seed=2,
                iteration=0,
            )
        self.assertEqual(fit_mock.call_count, 1)
        self.assertEqual(ddts.qubo_stats.num_variables, size)
        self.assertEqual(ddts.qubo_stats.system_penalty_weight, 650.0)

        captured: dict[str, object] = {}

        def capture_build(
            fm_q: np.ndarray,
            fm_bias: float,
            encoding: object,
            include_system_penalty: bool,
        ) -> object:
            captured["q"] = fm_q.copy()
            captured["bias"] = fm_bias
            captured["include_system_penalty"] = include_system_penalty
            return build_single_objective_qubo(fm_q, fm_bias, encoding, include_system_penalty)

        objective_qubos = [
            (np.eye(size) * 1.0, 1.0),
            (np.eye(size) * 2.0, 2.0),
            (np.eye(size) * 3.0, 3.0),
        ]
        with (
            mock.patch("figure5_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})) as fit_mock,
            mock.patch("figure5_pipeline.fm_to_qubo", side_effect=objective_qubos),
            mock.patch("figure5_pipeline.build_single_objective_qubo", side_effect=capture_build),
            mock.patch("figure5_pipeline.simulated_annealing_qubo", side_effect=fake_sa),
        ):
            weighted = pipeline._fit_and_solve_iteration(
                rows=rows,
                setting="wo_ddts",
                config=config,
                seed=2,
                iteration=0,
            )

        weights = preference_weights_for_iteration(2, 0)
        expected_scale = sum(weight * (index + 1) for index, weight in enumerate(weights))
        self.assertEqual(fit_mock.call_count, 3)
        self.assertTrue(np.allclose(captured["q"], np.eye(size) * expected_scale))
        self.assertAlmostEqual(float(captured["bias"]), expected_scale)
        self.assertTrue(captured["include_system_penalty"])
        self.assertEqual(weighted.scalarization_method, "weighted_sum")

    def test_candidate_decision_separates_proposed_and_replacement(self) -> None:
        config = _test_config(iterations=1)
        encoding = create_iteration_encoding(config.num_levels, seed=3, num_blocks=4)
        composition = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bits = encode_discrete_composition(composition, encoding)

        accepted = pipeline._decide_candidate(
            candidate_bits=bits,
            encoding=encoding,
            sample_id=10,
            seed=3,
            replacement_rng=random.Random(20),
            num_levels=config.num_levels,
            seen_compositions=set(),
        )
        self.assertEqual(accepted.status, "accepted")
        self.assertIs(accepted.proposed_row, accepted.added_row)

        duplicate = pipeline._decide_candidate(
            candidate_bits=bits,
            encoding=encoding,
            sample_id=10,
            seed=3,
            replacement_rng=random.Random(20),
            num_levels=config.num_levels,
            seen_compositions={pipeline._composition_key(composition)},
        )
        self.assertEqual(duplicate.status, "duplicate_replacement")
        self.assertIsNotNone(duplicate.proposed_row)
        self.assertNotEqual(
            pipeline._row_composition_key(duplicate.proposed_row),
            pipeline._row_composition_key(duplicate.added_row),
        )

        invalid = pipeline._decide_candidate(
            candidate_bits=np.zeros(4 * config.num_levels),
            encoding=encoding,
            sample_id=10,
            seed=3,
            replacement_rng=random.Random(21),
            num_levels=config.num_levels,
            seen_compositions=set(),
        )
        self.assertEqual(invalid.status, "invalid_replacement")
        self.assertIsNone(invalid.proposed_row)

    def test_partial_resume_and_completed_skip(self) -> None:
        config = _test_config(iterations=2)
        initial_rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=6)
        output_dir = WORKSPACE_TMP_ROOT / "figure5_resume"
        checkpoint = figure5_output_layout(output_dir).checkpoint_path("w_ddts", 6)

        with mock.patch(
            "figure5_pipeline._fit_and_solve_iteration",
            side_effect=[_training_result(config, 6, 0), RuntimeError("interrupt")],
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                run_single_trajectory(
                    initial_rows,
                    "w_ddts",
                    seed=6,
                    config=config,
                    checkpoint_path=checkpoint,
                    resume=False,
                )

        payload = load_checkpoint(checkpoint, setting="w_ddts", seed=6, config=config)
        self.assertEqual(len(payload["state"]["iteration_records"]), 1)

        with mock.patch(
            "figure5_pipeline._fit_and_solve_iteration",
            return_value=_training_result(config, 6, 1),
        ) as fit_mock:
            resumed = run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=6,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(fit_mock.call_count, 1)
        self.assertEqual(resumed.completed_iterations, 2)
        self.assertEqual(resumed.final_dataset_size, config.num_samples + config.iterations)
        self.assertEqual(resumed.invalid_replacements, 2)
        self.assertEqual(resumed.random_replacements, 2)
        self.assertGreaterEqual(resumed.random_replacement_draws, 2)

        with mock.patch("figure5_pipeline._fit_and_solve_iteration") as fit_mock:
            skipped = run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=6,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )
        fit_mock.assert_not_called()
        self.assertEqual(skipped.completed_iterations, 2)

        changed = _test_config(iterations=3)
        with self.assertRaisesRegex(ValueError, "config"):
            run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=6,
                config=changed,
                checkpoint_path=checkpoint,
                resume=True,
            )

    def test_summary_uses_proposed_solutions_and_excludes_replacements(self) -> None:
        config = _test_config(iterations=3)
        weak = SolutionPoint(10, [1.0, 0.0, 0.0, 0.0], 100.0, 70.0, 3.0)
        strong = SolutionPoint(11, [0.0, 1.0, 0.0, 0.0], 110.0, 80.0, 2.8)
        replacement = SolutionPoint(12, [0.0, 0.0, 1.0, 0.0], 90.0, 60.0, 3.2)

        def trajectory(setting: str, seed: int) -> Figure5TrajectoryResult:
            records = [
                IterationRecord(setting, seed, 0, [0.2, 0.3, 0.5], "ddts", "accepted", weak, weak, 0, -1.0),
                IterationRecord(
                    setting,
                    seed,
                    1,
                    [0.2, 0.3, 0.5],
                    "ddts",
                    "duplicate_replacement",
                    strong,
                    replacement,
                    1,
                    -2.0,
                ),
                IterationRecord(
                    setting,
                    seed,
                    2,
                    [0.2, 0.3, 0.5],
                    "ddts",
                    "invalid_replacement",
                    None,
                    replacement,
                    1,
                    -3.0,
                ),
            ]
            return Figure5TrajectoryResult(
                setting=setting,
                seed=seed,
                iteration_records=records,
                final_dataset_size=13,
                training_backend=pipeline.TRAINING_BACKEND,
                latest_weights=[0.2, 0.3, 0.5],
                latest_fm_metadata={"mock": True},
                latest_scalarization_metadata={"mock": True},
                qubo_stats=_qubo_stats(),
                duplicate_replacements=1,
                invalid_replacements=1,
                random_replacements=2,
                random_replacement_draws=2,
                accepted_sa_candidates=1,
                completed_iterations=3,
            )

        def fake_run(*_: object, **kwargs: object) -> Figure5TrajectoryResult:
            return trajectory(str(kwargs["setting"]), int(kwargs["seed"]))

        output_dir = WORKSPACE_TMP_ROOT / "figure5_summary"
        with mock.patch("figure5_pipeline.run_single_trajectory", side_effect=fake_run):
            summary = run_figure5_experiment(
                seed_list=[0],
                config=config,
                output_dir=output_dir,
                settings=("w_ddts", "wo_ddts"),
            )

        self.assertEqual(len(summary.solutions), 4)
        self.assertTrue(all(solution["composition"] != replacement.composition for solution in summary.solutions))
        self.assertEqual(len(summary.pareto_front["w_ddts"]), 1)
        self.assertEqual(summary.pareto_front["w_ddts"][0]["composition"], strong.composition)
        payload = json.loads(figure5_output_layout(output_dir).summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["settings"], ["w_ddts", "wo_ddts"])

    def test_output_checkpoint_and_manifest_schema(self) -> None:
        config = _test_config(iterations=1)
        output_dir = WORKSPACE_TMP_ROOT / "figure5_outputs"
        layout = figure5_output_layout(output_dir)
        state = pipeline.Figure5TrajectoryState(rows=[])
        checkpoint = layout.checkpoint_path("wo_ddts", 4)
        write_checkpoint(
            checkpoint,
            checkpoint_payload(setting="wo_ddts", seed=4, state=state, config=config),
        )
        loaded = load_checkpoint(checkpoint, setting="wo_ddts", seed=4, config=config)
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(checkpoint.name, "wo_ddts_seed_4.json")

        args = figure5_runner.parse_args(
            ["--output-dir", str(output_dir), "--preset", "test", "--settings", "w_ddts", "wo_ddts"]
        )
        manifest_path = figure5_runner._write_manifest(
            output_layout=layout,
            args=args,
            resolved_config=config,
            seed_list=[0],
            settings=["w_ddts", "wo_ddts"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["resolved_config"], config.to_dict())
        self.assertEqual(manifest["objectives"], ["kappa", "E", "rho"])
        self.assertEqual(manifest["settings"], ["w_ddts", "wo_ddts"])

    def test_runner_rejects_unavailable_cuda(self) -> None:
        with mock.patch("figure5_runner.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(EnvironmentError, "cuda"):
                figure5_runner._ensure_requested_device_available("cuda")


if __name__ == "__main__":
    unittest.main()
