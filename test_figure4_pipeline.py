import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from alloy_dataset_generator import build_dataset_row, generate_initial_dataset_single_objective
from figure4_config import OBJECTIVES, EncodingConfig, ExperimentConfig, FMConfig, SAConfig, resolve_experiment_config
from figure4_fm_torch import TorchFMRegressor, fm_to_qubo
from figure4_outputs import (
    CHECKPOINT_DIR_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FIGURE_FILENAME,
    LOG_DIR_NAME,
    MANIFEST_FILENAME,
    PLOT_LOG_FILENAME,
    RUNNER_LOG_FILENAME,
    SUMMARY_FILENAME,
    classify_output_path,
    configure_file_logging_path,
    checkpoint_payload,
    figure4_output_layout,
    load_checkpoint,
    load_summary,
    write_checkpoint,
)
from figure4_pipeline import (
    TrajectoryResult,
    aggregate_trajectory_results,
    discretize_rows_for_figure4,
    ensure_training_dependencies,
    run_figure4_experiment,
    run_single_trajectory,
)
from figure4_qubo_math import (
    QuboStats,
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
    validate_candidate_composition,
)
from figure4_settings import get_setting_strategy

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_test"
WORKSPACE_TMP_ROOT.mkdir(exist_ok=True)


class Figure4PipelineTests(unittest.TestCase):
    def test_required_runtime_dependencies_are_importable(self) -> None:
        ensure_training_dependencies()
        self.assertIsInstance(torch.__version__, str)
        self.assertTrue(torch.__version__)

    def test_config_validation_and_presets(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_levels"):
            EncodingConfig(num_levels=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            ExperimentConfig(iterations=0)
        with self.assertRaisesRegex(ValueError, "sa_runs"):
            SAConfig(runs=0, sweeps=1)
        with self.assertRaisesRegex(ValueError, "optuna_trials"):
            FMConfig(optuna_trials=-1)

        quick = resolve_experiment_config(preset="quick_l50", device="cuda")
        self.assertEqual(quick.num_samples, 100)
        self.assertEqual(quick.iterations, 100)
        self.assertEqual(quick.num_levels, 50)
        self.assertEqual(quick.optuna_trials, 3)
        self.assertEqual(quick.sa_runs, 100)
        self.assertEqual(quick.sa_sweeps, 500)
        self.assertEqual(quick.device, "cuda")

        overridden = resolve_experiment_config(preset="test", device="cpu", iterations=3, sa_runs=4)
        self.assertEqual(overridden.iterations, 3)
        self.assertEqual(overridden.sa_runs, 4)
        self.assertEqual(overridden.sa_sweeps, 6)

    def test_checkpoint_manager_rules(self) -> None:
        from figure4_pipeline import TrajectoryState

        config = ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        output_dir = WORKSPACE_TMP_ROOT / "checkpoint_rules"
        layout = figure4_output_layout(output_dir)
        path = layout.checkpoint_path("wo_cgfm", "kappa", 4)
        self.assertEqual(path, layout.checkpoint_path("wo_cgfm", "kappa", 4))
        self.assertEqual(path.name, "wo_cgfm_kappa_seed_4.json")
        self.assertEqual(path.parent, layout.checkpoint_dir)

        rows = [build_dataset_row(0, 4, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))]
        payload = checkpoint_payload(
            objective=OBJECTIVES[0],
            setting="wo_cgfm",
            seed=4,
            state=TrajectoryState(rows=rows),
            config=config,
        )
        write_checkpoint(path, payload)
        loaded = load_checkpoint(path, objective=OBJECTIVES[0], seed=4, config=config, setting="wo_cgfm")
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(loaded["objective"], "kappa")

        changed_config = ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=6),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        with self.assertRaisesRegex(ValueError, "config"):
            load_checkpoint(path, objective=OBJECTIVES[0], seed=4, config=changed_config, setting="wo_cgfm")

    def test_output_layout_and_path_classification_rules(self) -> None:
        output_dir = WORKSPACE_TMP_ROOT / "figure4_layout_rules"
        layout = figure4_output_layout(output_dir)
        self.assertEqual(layout.summary_path, output_dir / SUMMARY_FILENAME)
        self.assertEqual(layout.manifest_path, output_dir / MANIFEST_FILENAME)
        self.assertEqual(layout.figure_path, output_dir / DEFAULT_FIGURE_FILENAME)
        self.assertEqual(layout.runner_log_path, output_dir / LOG_DIR_NAME / RUNNER_LOG_FILENAME)
        self.assertEqual(layout.plot_log_path, output_dir / LOG_DIR_NAME / PLOT_LOG_FILENAME)
        self.assertEqual(
            layout.checkpoint_path("w_cgfm", "delta_T", 7),
            output_dir / CHECKPOINT_DIR_NAME / "w_cgfm_delta_T_seed_7.json",
        )

        self.assertEqual(classify_output_path(Path("figure4_quick_l50")), "protected_legacy")
        self.assertEqual(classify_output_path(Path("figure4_cgfm_quick_l50") / CHECKPOINT_DIR_NAME), "protected_legacy")
        self.assertEqual(classify_output_path(Path("figure4_compare_l50_from_separate")), "protected_legacy")
        self.assertEqual(classify_output_path(Path("figure4_quick_150")), "protected_legacy")
        self.assertEqual(classify_output_path(Path("figure4_quick150")), "protected_legacy")
        self.assertEqual(classify_output_path(WORKSPACE_TMP_ROOT), "temporary_test")
        self.assertEqual(classify_output_path(Path("figure4_refactor_test")), "temporary_test")
        self.assertEqual(classify_output_path(Path("figure4_compare_l50")), "managed_current")
        self.assertEqual(classify_output_path(Path("notes")), "unknown")

        summary_payload = {"schema_version": 2, "aggregated": {}}
        layout.summary_path.parent.mkdir(parents=True, exist_ok=True)
        layout.summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        self.assertEqual(load_summary(layout.summary_path), summary_payload)

        log_path = configure_file_logging_path(layout.runner_log_path)
        self.assertEqual(log_path, layout.runner_log_path)
        self.assertTrue(layout.runner_log_path.exists())

    def test_runner_checks_requested_cuda_availability(self) -> None:
        import figure4_runner

        with mock.patch("figure4_runner.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(EnvironmentError, "cuda"):
                figure4_runner._ensure_requested_device_available("cuda")
            figure4_runner._ensure_requested_device_available("cpu")

    def test_workflow_documentation_exists(self) -> None:
        repo_root = Path(__file__).resolve().parent
        document_path = repo_root / "README.md"
        content = document_path.read_text(encoding="utf-8")
        for heading in (
            "## 项目状态",
            "## 代码结构",
            "## Figure 4 算法流程",
            "## 配置规则",
            "## Figure 4 运行方法",
            "## 输出规则",
            "## 命名与 schema 规则",
            "## 返回类型与调用规则",
            "## Figure 5 复现工作流",
        ):
            self.assertIn(heading, content)
        for figure5_keyword in ("w_ddts", "wo_ddts", "weighted-sum", "Pareto front", "figure5_runner.py"):
            self.assertIn(figure5_keyword, content)
        legacy_document_name = "FIGURE4" + "_WORKFLOW.md"
        self.assertFalse((repo_root / legacy_document_name).exists())

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

    def test_strategy_dimensions_and_penalty_policy(self) -> None:
        wo_strategy = get_setting_strategy("wo_cgfm")
        w_strategy = get_setting_strategy("w_cgfm")
        self.assertTrue(wo_strategy.include_system_penalty)
        self.assertFalse(w_strategy.include_system_penalty)

        direct_encoding = wo_strategy.create_encoding(10, seed=1)
        cgfm_encoding = w_strategy.create_encoding(10, seed=1)
        self.assertEqual(direct_encoding.num_blocks, 4)
        self.assertEqual(cgfm_encoding.num_blocks, 3)

        direct_q = np.zeros((4 * direct_encoding.num_levels, 4 * direct_encoding.num_levels), dtype=np.float64)
        cgfm_q = np.zeros((3 * cgfm_encoding.num_levels, 3 * cgfm_encoding.num_levels), dtype=np.float64)
        direct_result = build_single_objective_qubo(direct_q, 0.0, direct_encoding, include_system_penalty=True)
        cgfm_result = build_single_objective_qubo(cgfm_q, 0.0, cgfm_encoding, include_system_penalty=False)

        self.assertEqual(direct_result.stats.num_variables, 40)
        self.assertEqual(direct_result.stats.system_penalty_weight, 650.0)
        self.assertEqual(cgfm_result.stats.num_variables, 30)
        self.assertEqual(cgfm_result.stats.system_penalty_weight, 0.0)

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

    def test_penalties_match_paper_zero_or_one_hot_semantics(self) -> None:
        encoding = create_iteration_encoding(num_levels=5, seed=7)
        system_q, system_bias = build_system_penalty_matrix(encoding)
        one_hot_q, one_hot_bias = build_one_hot_penalty_matrix(encoding)

        zero_block_row = build_dataset_row(0, 0, np.array([0.0, 0.2, 0.2, 0.6], dtype=np.float64))
        valid = encode_single_objective_rows([zero_block_row], encoding)[0].astype(np.float64)
        decoded_valid = decode_candidate_bits_to_composition(valid, encoding)
        self.assertIsNotNone(decoded_valid)
        self.assertTrue(validate_candidate_composition(decoded_valid))
        self.assertAlmostEqual(evaluate_qubo_energy(valid, one_hot_q, one_hot_bias), 0.0, places=12)

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
        self.assertLess(evaluate_qubo_energy(feasible, system_q, system_bias), evaluate_qubo_energy(infeasible, system_q, system_bias))

    def test_run_single_trajectory_grows_dataset(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=12, seed=3)
        config = ExperimentConfig(
            num_samples=12,
            iterations=2,
            encoding=EncodingConfig(num_levels=10),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=2, sweeps=6),
        )
        result = run_single_trajectory(rows, OBJECTIVES[0], seed=3, config=config, setting="wo_cgfm")
        self.assertEqual(len(result.best_so_far), 2)
        self.assertEqual(result.final_dataset_size, 14)
        initial_best = max(
            float(
                build_dataset_row(
                    int(row["sample_id"]),
                    int(row["seed"]),
                    prepare_discrete_composition(
                        np.array([row["f1_norm"], row["f2_norm"], row["f3_norm"], row["f4_norm"]], dtype=np.float64),
                        10,
                    ),
                )["kappa"]
            )
            for row in rows
        )
        self.assertGreaterEqual(result.best_so_far[0], initial_best)

    def test_run_single_cgfm_trajectory_creates_valid_result(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=8, seed=7)
        config = ExperimentConfig(
            num_samples=8,
            iterations=1,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=4),
        )
        q = np.zeros((3 * config.num_levels, 3 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(3 * config.num_levels), 0.0)),
        ):
            result = run_single_trajectory(rows, OBJECTIVES[0], seed=7, config=config, setting="w_cgfm")
        self.assertEqual(result.setting, "w_cgfm")
        self.assertEqual(result.final_dataset_size, 9)
        self.assertEqual(result.completed_iterations, 1)
        self.assertIsNotNone(result.qubo_stats)
        self.assertEqual(result.qubo_stats.num_variables, 24)
        self.assertEqual(result.qubo_stats.system_penalty_weight, 0.0)

    def test_aggregate_includes_best_so_far_dispersion(self) -> None:
        results = [
            TrajectoryResult("kappa", "wo_cgfm", 0, [1.0, 3.0, 5.0], 3, "mock", {}, completed_iterations=3),
            TrajectoryResult("kappa", "wo_cgfm", 1, [3.0, 5.0, 7.0], 3, "mock", {}, completed_iterations=3),
        ]
        aggregated = aggregate_trajectory_results(results)
        stats = aggregated["kappa:wo_cgfm"]
        self.assertEqual(stats.best_so_far_mean, [2.0, 4.0, 6.0])
        self.assertEqual(stats.best_so_far_min, [1.0, 3.0, 5.0])
        self.assertEqual(stats.best_so_far_max, [3.0, 5.0, 7.0])
        self.assertEqual(stats.best_so_far_std, [1.0, 1.0, 1.0])
        self.assertEqual(stats.num_trajectories, 2)

    def test_invalid_and_duplicate_candidates_are_counted_as_replacements(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=8)
        config = ExperimentConfig(
            num_samples=6,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)

        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
        ):
            invalid_result = run_single_trajectory(rows, OBJECTIVES[0], seed=8, config=config, setting="wo_cgfm")
        self.assertEqual(invalid_result.invalid_replacements, 1)
        self.assertEqual(invalid_result.duplicate_replacements, 0)
        self.assertEqual(invalid_result.accepted_sa_candidates, 0)
        self.assertEqual(invalid_result.random_replacements, 1)

        duplicate_composition = discretize_rows_for_figure4(rows, config.num_levels)[0]
        duplicate_array = np.array(
            [
                duplicate_composition["f1_norm"],
                duplicate_composition["f2_norm"],
                duplicate_composition["f3_norm"],
                duplicate_composition["f4_norm"],
            ],
            dtype=np.float64,
        )
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
            mock.patch("figure4_settings.WOCGFMStrategy.decode_candidate", return_value=duplicate_array),
        ):
            duplicate_result = run_single_trajectory(rows, OBJECTIVES[0], seed=8, config=config, setting="wo_cgfm")
        self.assertEqual(duplicate_result.invalid_replacements, 0)
        self.assertEqual(duplicate_result.duplicate_replacements, 1)
        self.assertEqual(duplicate_result.accepted_sa_candidates, 0)
        self.assertEqual(duplicate_result.random_replacements, 1)

    def test_random_replacement_is_forced_to_be_novel(self) -> None:
        duplicate = np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64)
        unique = np.array([0.0, 0.2, 0.2, 0.6], dtype=np.float64)
        rows = [
            build_dataset_row(0, 11, duplicate),
            build_dataset_row(1, 11, np.array([0.4, 0.2, 0.2, 0.2], dtype=np.float64)),
        ]
        config = ExperimentConfig(
            num_samples=2,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        output_dir = WORKSPACE_TMP_ROOT / "novel_replacement"
        checkpoint = figure4_output_layout(output_dir).checkpoint_path("wo_cgfm", "kappa", 11)
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
            mock.patch("figure4_pipeline.sample_single_objective_design", side_effect=[tuple(duplicate), tuple(unique)]),
        ):
            result = run_single_trajectory(
                rows,
                OBJECTIVES[0],
                seed=11,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
            )
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        final_row = payload["state"]["rows"][-1]
        self.assertEqual(result.random_replacements, 1)
        self.assertEqual(payload["state"]["random_replacement_draws"], 2)
        self.assertEqual(
            [final_row["f1_norm"], final_row["f2_norm"], final_row["f3_norm"], final_row["f4_norm"]],
            unique.tolist(),
        )

    def test_resume_skips_completed_and_continues_partial_checkpoint(self) -> None:
        from figure4_pipeline import TrajectoryState

        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=9)
        config = ExperimentConfig(
            num_samples=6,
            iterations=3,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        output_dir = WORKSPACE_TMP_ROOT / "resume_output_v2"
        checkpoint = figure4_output_layout(output_dir).checkpoint_path("wo_cgfm", "kappa", 9)
        prepared_rows = discretize_rows_for_figure4(rows, config.num_levels)
        replacement_row = build_dataset_row(len(prepared_rows), 9, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        partial_rows = [*prepared_rows, replacement_row]
        partial_best = max(float(row["kappa"]) for row in partial_rows)
        state = TrajectoryState(
            rows=partial_rows,
            best_so_far=[partial_best],
            fm_metadata={"mock": True},
            qubo_stats=QuboStats(1.0, 1.0, 1.0, 650.0, 1.0, 20, 650.0),
            invalid_replacements=1,
            random_replacements=1,
            random_replacement_draws=1,
        )
        payload = checkpoint_payload(
            objective=OBJECTIVES[0],
            setting="wo_cgfm",
            seed=9,
            state=state,
            config=config,
        )
        write_checkpoint(checkpoint, payload)
        loaded = load_checkpoint(
            checkpoint,
            objective=OBJECTIVES[0],
            seed=9,
            config=config,
            setting="wo_cgfm",
        )
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch("figure4_pipeline.simulated_annealing_qubo", return_value=(np.zeros(4 * config.num_levels), 0.0)),
        ):
            resumed = run_single_trajectory(
                rows,
                OBJECTIVES[0],
                seed=9,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
                resume=True,
            )
            skipped = run_single_trajectory(
                rows,
                OBJECTIVES[0],
                seed=9,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(resumed.completed_iterations, 3)
        self.assertEqual(len(resumed.best_so_far), 3)
        self.assertEqual(resumed.final_dataset_size, config.num_samples + config.iterations)
        self.assertEqual(resumed.invalid_replacements, 3)
        self.assertEqual(skipped.completed_iterations, 3)

    def test_checkpoint_config_mismatch_fails(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=4, seed=12)
        config = ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        output_dir = WORKSPACE_TMP_ROOT / "config_mismatch_v2"
        run_figure4_experiment(
            seed_list=[12],
            config=config,
            output_dir=output_dir,
            objectives=(OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm",),
        )
        changed_config = ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=6),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=1),
        )
        with self.assertRaisesRegex(ValueError, "config"):
            run_single_trajectory(
                rows,
                OBJECTIVES[0],
                seed=12,
                config=changed_config,
                setting="wo_cgfm",
                checkpoint_path=figure4_output_layout(output_dir).checkpoint_path("wo_cgfm", "kappa", 12),
                resume=True,
            )

    def test_experiment_summary_and_plot_generation(self) -> None:
        config = ExperimentConfig(
            num_samples=10,
            iterations=2,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=2, sweeps=6),
        )
        output_dir = WORKSPACE_TMP_ROOT / "figure4_output_v2"
        summary = run_figure4_experiment(
            seed_list=[0],
            config=config,
            output_dir=output_dir,
            objectives=(OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm", "w_cgfm"),
        )
        layout = figure4_output_layout(output_dir)
        summary_path = layout.summary_path
        output_png = layout.figure_path
        self.assertTrue(summary_path.exists())
        self.assertEqual(summary.schema_version, 2)
        self.assertEqual(summary.training_backend, "pytorch_fm_lbfgs")
        self.assertIn("kappa:wo_cgfm", summary.aggregated)
        self.assertIn("kappa:w_cgfm", summary.aggregated)
        self.assertTrue(summary.aggregated["kappa:wo_cgfm"].best_so_far_std)
        subprocess.run(
            [sys.executable, "plot_figure4.py", "--summary", str(summary_path), "--output", str(output_png)],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
        )
        self.assertTrue(output_png.exists())
        self.assertTrue(layout.plot_log_path.exists())

    def test_plot_accepts_multiple_summaries(self) -> None:
        config = ExperimentConfig(
            num_samples=8,
            iterations=1,
            encoding=EncodingConfig(num_levels=6),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(runs=1, sweeps=4),
        )
        wo_dir = WORKSPACE_TMP_ROOT / "figure4_multi_plot_wo"
        w_dir = WORKSPACE_TMP_ROOT / "figure4_multi_plot_w"
        run_figure4_experiment(
            seed_list=[1],
            config=config,
            output_dir=wo_dir,
            objectives=(OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm",),
        )
        run_figure4_experiment(
            seed_list=[1],
            config=config,
            output_dir=w_dir,
            objectives=(OBJECTIVES[0],),
            resume=True,
            settings=("w_cgfm",),
        )
        output_png = WORKSPACE_TMP_ROOT / "figure4_multi_plot.png"
        subprocess.run(
            [
                sys.executable,
                "plot_figure4.py",
                "--summary",
                str(figure4_output_layout(wo_dir).summary_path),
                str(figure4_output_layout(w_dir).summary_path),
                "--output",
                str(output_png),
            ],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
        )
        self.assertTrue(output_png.exists())

    def test_plot_rejects_unsupported_or_empty_summary(self) -> None:
        invalid_dir = WORKSPACE_TMP_ROOT / "figure4_invalid_plot"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        old_schema_summary = invalid_dir / "old_schema.json"
        old_schema_summary.write_text(
            json.dumps({"schema_version": 1, "aggregated": {"kappa:wo_cgfm": {}}}),
            encoding="utf-8",
        )
        empty_summary = invalid_dir / "empty_summary.json"
        empty_summary.write_text(json.dumps({"schema_version": 2, "aggregated": {}}), encoding="utf-8")

        for summary_path in (old_schema_summary, empty_summary):
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        "plot_figure4.py",
                        "--summary",
                        str(summary_path),
                        "--output",
                        str(invalid_dir / f"{summary_path.stem}.png"),
                    ],
                    cwd=str(Path(__file__).resolve().parent),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    unittest.main()
