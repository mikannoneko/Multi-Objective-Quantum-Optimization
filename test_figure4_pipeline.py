import json
import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from alloy_dataset_generator import build_dataset_row, generate_initial_dataset_single_objective
from figure4_pipeline import (
    Figure4Config,
    OBJECTIVES,
    TrajectoryResult,
    TorchFMRegressor,
    aggregate_trajectory_results,
    build_one_hot_penalty_matrix,
    build_system_penalty_matrix,
    build_single_objective_qubo,
    cgfm_angles_to_composition,
    cgfm_composition_to_angles,
    create_cgfm_iteration_encoding,
    create_iteration_encoding,
    decode_candidate_bits_to_cgfm_composition,
    decode_candidate_bits_to_composition,
    encode_cgfm_rows,
    encode_single_objective_rows,
    ensure_runtime_dependencies,
    evaluate_qubo_energy,
    fm_to_qubo,
    prepare_discrete_composition,
    prepare_paper_rows,
    run_figure4_experiment,
    run_single_trajectory,
    validate_candidate_composition,
)

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_test"
WORKSPACE_TMP_ROOT.mkdir(exist_ok=True)


class Figure4PipelineTests(unittest.TestCase):
    def test_required_runtime_dependencies_are_importable(self) -> None:
        ensure_runtime_dependencies()
        self.assertEqual(torch.__version__, "1.9.1+cu111")

    def test_torch_fm_forward_shape(self) -> None:
        model = TorchFMRegressor(num_features=8, init_std=0.1)
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1,))

    def test_fm_to_qubo_matches_forward_on_binary_inputs(self) -> None:
        model = TorchFMRegressor(num_features=5, init_std=0.1)
        with torch.no_grad():
            model.w0.copy_(torch.tensor([0.7]))
            model.w.copy_(torch.tensor([0.2, -0.1, 0.4, 0.0, 0.3]))
            model.V.copy_(
                torch.tensor(
                    [
                        [0.5, -0.2, 0.1, 0.3, -0.1, 0.0],
                        [0.1, 0.3, -0.2, 0.4, 0.5, -0.2],
                        [-0.4, 0.2, 0.0, 0.1, 0.4, 0.3],
                        [0.0, 0.6, 0.3, -0.2, 0.1, 0.2],
                        [0.3, -0.1, 0.2, 0.0, -0.3, 0.4],
                    ],
                    dtype=torch.float32,
                )
            )
        q, bias = fm_to_qubo(model)
        rng = np.random.default_rng(123)
        for _ in range(20):
            bits = rng.integers(0, 2, size=5).astype(np.float32)
            fm_value = float(model(torch.from_numpy(bits[None, :])).item())
            qubo_value = evaluate_qubo_energy(bits, q, bias)
            self.assertAlmostEqual(fm_value, qubo_value, places=5)

    def test_zero_value_uses_all_zero_block(self) -> None:
        encoding = create_iteration_encoding(num_levels=10, seed=5)
        row = build_dataset_row(0, 0, np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64))
        features = encode_single_objective_rows([row], encoding)
        self.assertTrue(np.all(features[0, :10] == 0.0))

    def test_bit_swap_curing_enforces_system_constraint(self) -> None:
        continuous = np.array([0.333, 0.333, 0.333, 0.001], dtype=np.float64)
        discrete = prepare_discrete_composition(continuous, num_levels=50)
        self.assertAlmostEqual(float(np.sum(discrete)), 1.0, places=12)
        scaled = discrete * 50
        self.assertTrue(np.allclose(scaled, np.rint(scaled)))

    def test_iteration_encoding_shuffle_is_reproducible_and_decodable(self) -> None:
        encoding_a = create_iteration_encoding(10, seed=11)
        encoding_b = create_iteration_encoding(10, seed=11)
        encoding_c = create_iteration_encoding(10, seed=12)
        for block_idx in range(4):
            self.assertTrue(np.array_equal(encoding_a.positive_count_by_bit[block_idx], encoding_b.positive_count_by_bit[block_idx]))
        self.assertTrue(
            any(
                not np.array_equal(encoding_a.positive_count_by_bit[idx], encoding_c.positive_count_by_bit[idx])
                for idx in range(4)
            )
        )

        row = build_dataset_row(0, 0, np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64))
        bits = encode_single_objective_rows([row], encoding_a)[0]
        decoded = decode_candidate_bits_to_composition(bits, encoding_a)
        self.assertIsNotNone(decoded)
        self.assertTrue(np.allclose(decoded, np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64)))

    def test_cgfm_mapping_roundtrip_and_feature_shape(self) -> None:
        composition = np.array([0.12, 0.28, 0.35, 0.25], dtype=np.float64)
        encoding_a = create_cgfm_iteration_encoding(10, seed=11)
        encoding_b = create_cgfm_iteration_encoding(10, seed=11)
        encoding_c = create_cgfm_iteration_encoding(10, seed=12)
        self.assertEqual(encoding_a.num_blocks, 3)
        self.assertTrue(np.array_equal(encoding_a.phase_permutation, encoding_b.phase_permutation))
        self.assertFalse(np.array_equal(encoding_a.phase_permutation, encoding_c.phase_permutation))

        angles = cgfm_composition_to_angles(composition, encoding_a.phase_permutation)
        decoded = cgfm_angles_to_composition(angles, encoding_a.phase_permutation)
        self.assertTrue(np.all(decoded >= -1e-12))
        self.assertAlmostEqual(float(np.sum(decoded)), 1.0, places=12)
        self.assertTrue(np.allclose(decoded, composition, atol=1e-10))

        row = build_dataset_row(0, 0, composition)
        features = encode_cgfm_rows([row], encoding_a)
        self.assertEqual(features.shape, (1, 30))

    def test_cgfm_decode_enforces_one_hot_and_sum_constraint(self) -> None:
        encoding = create_cgfm_iteration_encoding(5, seed=13)
        bits = np.zeros(3 * encoding.num_levels, dtype=np.float64)
        decoded = decode_candidate_bits_to_cgfm_composition(bits, encoding)
        self.assertIsNotNone(decoded)
        self.assertTrue(validate_candidate_composition(decoded))

        invalid = bits.copy()
        invalid[0] = 1.0
        invalid[1] = 1.0
        self.assertIsNone(decode_candidate_bits_to_cgfm_composition(invalid, encoding))

    def test_cgfm_qubo_omits_system_penalty(self) -> None:
        direct_encoding = create_iteration_encoding(10, seed=1)
        cgfm_encoding = create_cgfm_iteration_encoding(10, seed=1)
        direct_q = np.zeros((4 * direct_encoding.num_levels, 4 * direct_encoding.num_levels), dtype=np.float64)
        cgfm_q = np.zeros((3 * cgfm_encoding.num_levels, 3 * cgfm_encoding.num_levels), dtype=np.float64)

        _, _, direct_stats = build_single_objective_qubo(direct_q, 0.0, direct_encoding, include_system_penalty=True)
        _, _, cgfm_stats = build_single_objective_qubo(cgfm_q, 0.0, cgfm_encoding, include_system_penalty=False)

        self.assertEqual(direct_stats["num_variables"], 40.0)
        self.assertEqual(direct_stats["system_penalty_weight"], 650.0)
        self.assertEqual(cgfm_stats["num_variables"], 30.0)
        self.assertEqual(cgfm_stats["system_penalty_weight"], 0.0)

    def test_penalties_match_paper_zero_or_one_hot_semantics(self) -> None:
        encoding = create_iteration_encoding(num_levels=5, seed=7)
        system_q, system_bias = build_system_penalty_matrix(encoding)
        one_hot_q, one_hot_bias = build_one_hot_penalty_matrix(encoding)

        zero_block_row = build_dataset_row(0, 0, np.array([0.0, 0.2, 0.2, 0.6], dtype=np.float64))
        valid = encode_single_objective_rows([zero_block_row], encoding)[0].astype(np.float64)
        self.assertIsNotNone(decode_candidate_bits_to_composition(valid, encoding))
        self.assertTrue(validate_candidate_composition(decode_candidate_bits_to_composition(valid, encoding)))

        legal_zero_block = valid.copy()
        self.assertAlmostEqual(evaluate_qubo_energy(legal_zero_block, one_hot_q, one_hot_bias), 0.0, places=12)

        invalid_one_hot = valid.copy()
        first_active = int(np.flatnonzero(invalid_one_hot > 0.5)[0])
        block_start = (first_active // encoding.num_levels) * encoding.num_levels
        invalid_index = block_start + ((first_active + 1) % encoding.num_levels)
        if invalid_index == first_active:
            invalid_index = block_start + ((first_active + 2) % encoding.num_levels)
        invalid_one_hot[invalid_index] = 1.0
        self.assertIsNone(decode_candidate_bits_to_composition(invalid_one_hot, encoding))
        self.assertGreater(evaluate_qubo_energy(invalid_one_hot, one_hot_q, one_hot_bias), 0.0)

        infeasible = valid.copy()
        infeasible[np.flatnonzero(infeasible > 0.5)[-1]] = 0.0
        decoded_infeasible = decode_candidate_bits_to_composition(infeasible, encoding)
        self.assertIsNotNone(decoded_infeasible)
        self.assertFalse(validate_candidate_composition(decoded_infeasible))

        feasible_row = build_dataset_row(1, 0, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        feasible = encode_single_objective_rows([feasible_row], encoding)[0].astype(np.float64)
        decoded_feasible = decode_candidate_bits_to_composition(feasible, encoding)
        self.assertIsNotNone(decoded_feasible)
        self.assertTrue(validate_candidate_composition(decoded_feasible))
        self.assertLess(evaluate_qubo_energy(feasible, system_q, system_bias), evaluate_qubo_energy(infeasible, system_q, system_bias))

    def test_run_single_trajectory_grows_dataset(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=12, seed=3)
        config = Figure4Config(
            num_samples=12,
            iterations=3,
            num_levels=10,
            optuna_trials=2,
            sa_runs=3,
            sa_sweeps=8,
        )
        result = run_single_trajectory(rows, OBJECTIVES[0], objective_index=0, seed=3, config=config, setting="wo_cgfm")
        self.assertEqual(len(result.history_best), 3)
        self.assertEqual(result.final_dataset_size, 15)
        initial_best = max(float(build_dataset_row(int(row["sample_id"]), int(row["seed"]), prepare_discrete_composition(
            np.array([row["f1_norm"], row["f2_norm"], row["f3_norm"], row["f4_norm"]], dtype=np.float64),
            10,
        ))["kappa"]) for row in rows)
        self.assertGreaterEqual(result.history_best[0], initial_best)

    def test_run_single_cgfm_trajectory_creates_valid_result(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=8, seed=7)
        config = Figure4Config(num_samples=8, iterations=1, num_levels=8, optuna_trials=0, sa_runs=1, sa_sweeps=4)
        q = np.zeros((3 * config.num_levels, 3 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(3 * config.num_levels), 0.0)),
        ):
            result = run_single_trajectory(rows, OBJECTIVES[0], objective_index=0, seed=7, config=config, setting="w_cgfm")
        self.assertEqual(result.setting, "w_cgfm")
        self.assertEqual(result.final_dataset_size, 9)
        self.assertEqual(result.completed_iterations, 1)
        self.assertEqual(result.qubo_stats["num_variables"], 24.0)
        self.assertEqual(result.qubo_stats["system_penalty_weight"], 0.0)

    def test_aggregate_includes_dispersion_curves(self) -> None:
        results = [
            TrajectoryResult("kappa", "wo_cgfm", 0, [1.0, 3.0, 5.0], 3, "mock", {}, completed_iterations=3),
            TrajectoryResult("kappa", "wo_cgfm", 1, [3.0, 5.0, 7.0], 3, "mock", {}, completed_iterations=3),
        ]
        aggregated = aggregate_trajectory_results(results)
        stats = aggregated["kappa:wo_cgfm"]
        self.assertEqual(stats["mean_curve"], [2.0, 4.0, 6.0])
        self.assertEqual(stats["min_curve"], [1.0, 3.0, 5.0])
        self.assertEqual(stats["max_curve"], [3.0, 5.0, 7.0])
        self.assertEqual(stats["std_curve"], [1.0, 1.0, 1.0])
        self.assertEqual(stats["num_trajectories"], 2)

    def test_invalid_and_duplicate_candidates_are_counted_as_replacements(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=8)
        config = Figure4Config(num_samples=6, iterations=1, num_levels=5, optuna_trials=0, sa_runs=1, sa_sweeps=1)
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)

        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
        ):
            invalid_result = run_single_trajectory(rows, OBJECTIVES[0], 0, seed=8, config=config, setting="wo_cgfm")
        self.assertEqual(invalid_result.invalid_replacements, 1)
        self.assertEqual(invalid_result.duplicate_replacements, 0)
        self.assertEqual(invalid_result.accepted_candidates, 0)

        duplicate_composition = prepare_paper_rows(rows, config.num_levels)[0]
        duplicate_array = np.array(
            [duplicate_composition["f1_norm"], duplicate_composition["f2_norm"], duplicate_composition["f3_norm"], duplicate_composition["f4_norm"]],
            dtype=np.float64,
        )
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
            mock.patch("figure4_pipeline.decode_candidate_bits_to_composition", return_value=duplicate_array),
        ):
            duplicate_result = run_single_trajectory(rows, OBJECTIVES[0], 0, seed=8, config=config, setting="wo_cgfm")
        self.assertEqual(duplicate_result.invalid_replacements, 0)
        self.assertEqual(duplicate_result.duplicate_replacements, 1)
        self.assertEqual(duplicate_result.accepted_candidates, 0)

    def test_resume_skips_completed_and_continues_partial_checkpoint(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=9)
        config = Figure4Config(num_samples=6, iterations=3, num_levels=5, optuna_trials=0, sa_runs=1, sa_sweeps=1)
        output_dir = WORKSPACE_TMP_ROOT / "resume_output"
        checkpoint = output_dir / "trajectories" / "wo_cgfm_kappa_seed_9.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        prepared_rows = prepare_paper_rows(rows, config.num_levels)
        replacement_row = build_dataset_row(len(prepared_rows), 9, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        partial_rows = [*prepared_rows, replacement_row]
        partial_best = max(float(row["kappa"]) for row in partial_rows)
        checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "objective": "kappa",
                    "setting": "wo_cgfm",
                    "seed": 9,
                    "history_best": [partial_best],
                    "final_dataset_size": len(partial_rows),
                    "training_backend": "pytorch_fm_lbfgs",
                    "fm_hparams": {"mock": True},
                    "qubo_stats": {"mock": True},
                    "duplicate_replacements": 0,
                    "invalid_replacements": 1,
                    "accepted_candidates": 0,
                    "completed_iterations": 1,
                    "config": asdict(config),
                    "rows": partial_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
        ):
            resumed = run_single_trajectory(
                rows,
                OBJECTIVES[0],
                objective_index=0,
                seed=9,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
                resume=True,
            )
            skipped = run_single_trajectory(
                rows,
                OBJECTIVES[0],
                objective_index=0,
                seed=9,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(resumed.completed_iterations, 3)
        self.assertEqual(len(resumed.history_best), 3)
        self.assertEqual(resumed.final_dataset_size, config.num_samples + config.iterations)
        self.assertEqual(resumed.invalid_replacements, 3)
        self.assertEqual(skipped.completed_iterations, 3)

    def test_legacy_wo_cgfm_checkpoint_is_copied_to_new_name(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=10)
        config = Figure4Config(num_samples=6, iterations=1, num_levels=5, optuna_trials=0, sa_runs=1, sa_sweeps=1)
        output_dir = WORKSPACE_TMP_ROOT / "legacy_checkpoint_output"
        legacy_checkpoint = output_dir / "trajectories" / "kappa_seed_10.json"
        new_checkpoint = output_dir / "trajectories" / "wo_cgfm_kappa_seed_10.json"
        legacy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        prepared_rows = prepare_paper_rows(rows, config.num_levels)
        partial_best = max(float(row["kappa"]) for row in prepared_rows)
        legacy_checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "objective": "kappa",
                    "setting": "wo_cgfm",
                    "seed": 10,
                    "history_best": [partial_best],
                    "final_dataset_size": len(prepared_rows),
                    "training_backend": "pytorch_fm_lbfgs",
                    "fm_hparams": {"mock": True},
                    "qubo_stats": {"mock": True},
                    "duplicate_replacements": 0,
                    "invalid_replacements": 0,
                    "accepted_candidates": 1,
                    "completed_iterations": 1,
                    "config": asdict(config),
                    "rows": prepared_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = run_figure4_experiment(
            seed_list=[10],
            config=config,
            output_dir=output_dir,
            objectives=(OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm",),
        )
        self.assertTrue(legacy_checkpoint.exists())
        self.assertTrue(new_checkpoint.exists())
        self.assertEqual(summary["trajectories"][0]["completed_iterations"], 1)

    def test_experiment_summary_and_plot_generation(self) -> None:
        config = Figure4Config(
            num_samples=10,
            iterations=2,
            num_levels=8,
            optuna_trials=0,
            sa_runs=2,
            sa_sweeps=6,
        )
        output_dir = WORKSPACE_TMP_ROOT / "figure4_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = run_figure4_experiment(
            seed_list=[0],
            config=config,
            output_dir=output_dir,
            objectives=(OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm", "w_cgfm"),
        )
        summary_path = output_dir / "figure4_summary.json"
        output_png = output_dir / "figure4.png"
        self.assertTrue(summary_path.exists())
        self.assertEqual(summary["training_backend"], "pytorch_fm_lbfgs")
        self.assertIn("std_curve", summary["aggregated"]["kappa:wo_cgfm"])
        self.assertIn("std_curve", summary["aggregated"]["kappa:w_cgfm"])
        subprocess.run(
            [sys.executable, "plot_figure4.py", "--summary", str(summary_path), "--output", str(output_png)],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
        )
        self.assertTrue(output_png.exists())


if __name__ == "__main__":
    unittest.main()
