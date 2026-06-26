import csv
import random
import unittest
from pathlib import Path

from alloy_dataset_generator import (
    AL_MATRIX_VOLUME_FRACTION,
    CSV_FIELDNAMES,
    DEFAULT_MULTI_OBJECTIVE_NUM_SAMPLES,
    compute_dataset_statistics,
    compute_density,
    compute_properties,
    generate_initial_dataset_multi_objective,
    generate_initial_dataset_single_objective,
    normalized_to_volume_fractions,
    sample_multi_objective_design,
    sample_single_objective_design,
)

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_test"
WORKSPACE_TMP_ROOT.mkdir(exist_ok=True)


class AlloyDatasetGeneratorTests(unittest.TestCase):
    def test_sampling_produces_feasible_normalized_compositions(self) -> None:
        rng = random.Random(123)
        for _ in range(200):
            design = sample_single_objective_design(rng)
            self.assertTrue(all(value >= 0.0 for value in design))
            self.assertAlmostEqual(sum(design), 1.0, places=12)

    def test_multi_objective_sampling_produces_feasible_normalized_compositions(self) -> None:
        rng = random.Random(456)
        for _ in range(200):
            design = sample_multi_objective_design(rng)
            self.assertEqual(len(design), 4)
            self.assertTrue(all(value >= 0.0 for value in design))
            self.assertAlmostEqual(sum(design), 1.0, places=12)

    def test_generated_rows_preserve_volume_balance(self) -> None:
        rows, stats = generate_initial_dataset_single_objective(num_samples=100, seed=7)
        self.assertEqual(len(rows), 100)
        self.assertLessEqual(float(stats["max_sum_deviation"]), 1e-12)

        for row in rows:
            total_norm = sum(float(row[field]) for field in ("f1_norm", "f2_norm", "f3_norm", "f4_norm"))
            total_secondary = sum(float(row[field]) for field in ("f1_vol", "f2_vol", "f3_vol", "f4_vol"))
            self.assertAlmostEqual(total_norm, 1.0, places=12)
            self.assertAlmostEqual(total_secondary, 0.2, places=12)
            self.assertAlmostEqual(float(row["f_Al_vol"]) + total_secondary, 1.0, places=12)
            self.assertGreaterEqual(float(row["delta_alpha"]), 0.0)
            self.assertGreaterEqual(float(row["delta_T"]), 0.0)

    def test_multi_objective_generated_rows_preserve_volume_balance(self) -> None:
        rows, stats = generate_initial_dataset_multi_objective(seed=7)
        self.assertEqual(len(rows), DEFAULT_MULTI_OBJECTIVE_NUM_SAMPLES)
        self.assertEqual(stats["num_samples"], DEFAULT_MULTI_OBJECTIVE_NUM_SAMPLES)
        self.assertLessEqual(float(stats["max_sum_deviation"]), 1e-12)

        for row in rows:
            total_norm = sum(float(row[field]) for field in ("f1_norm", "f2_norm", "f3_norm", "f4_norm"))
            total_secondary = sum(float(row[field]) for field in ("f1_vol", "f2_vol", "f3_vol", "f4_vol"))
            self.assertAlmostEqual(total_norm, 1.0, places=12)
            self.assertAlmostEqual(total_secondary, 0.2, places=12)
            self.assertAlmostEqual(float(row["f_Al_vol"]) + total_secondary, 1.0, places=12)
            self.assertGreaterEqual(float(row["delta_alpha"]), 0.0)
            self.assertGreaterEqual(float(row["delta_T"]), 0.0)

    def test_density_matches_weighted_average(self) -> None:
        normalized = (0.25, 0.25, 0.25, 0.25)
        volume = normalized_to_volume_fractions(normalized)
        expected = (0.8 * 2.7) + (0.05 * 2.3) + (0.05 * 2.0) + (0.05 * 4.0) + (0.05 * 4.4)
        self.assertAlmostEqual(compute_density(volume), expected, places=12)

    def test_property_computation_stays_finite_for_extreme_designs(self) -> None:
        extreme_designs = [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
        for normalized in extreme_designs:
            volume = normalized_to_volume_fractions(normalized)
            props = compute_properties(volume)
            for value in props.values():
                self.assertTrue(value == value)  # NaN check
                self.assertNotEqual(value, float("inf"))
                self.assertNotEqual(value, float("-inf"))

    def test_generation_is_deterministic_and_csv_is_written(self) -> None:
        rows_a, stats_a = generate_initial_dataset_single_objective(num_samples=10, seed=11)
        rows_b, stats_b = generate_initial_dataset_single_objective(num_samples=10, seed=11)
        rows_c, _ = generate_initial_dataset_single_objective(num_samples=10, seed=12)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(stats_a, stats_b)
        self.assertNotEqual(rows_a, rows_c)

        output_dir = WORKSPACE_TMP_ROOT / "dataset_generator_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset.csv"
        generate_initial_dataset_single_objective(num_samples=5, seed=22, output_path=output_path)
        self.assertTrue(output_path.exists())
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, CSV_FIELDNAMES)
            rows = list(reader)
        self.assertEqual(len(rows), 5)

    def test_multi_objective_generation_is_deterministic_and_csv_is_written(self) -> None:
        rows_a, stats_a = generate_initial_dataset_multi_objective(num_samples=10, seed=11)
        rows_b, stats_b = generate_initial_dataset_multi_objective(num_samples=10, seed=11)
        rows_c, _ = generate_initial_dataset_multi_objective(num_samples=10, seed=12)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(stats_a, stats_b)
        self.assertNotEqual(rows_a, rows_c)

        output_dir = WORKSPACE_TMP_ROOT / "multi_objective_dataset_generator_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset.csv"
        generate_initial_dataset_multi_objective(num_samples=5, seed=22, output_path=output_path)
        self.assertTrue(output_path.exists())
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, CSV_FIELDNAMES)
            rows = list(reader)
        self.assertEqual(len(rows), 5)

    def test_statistics_shape_matches_rows(self) -> None:
        rows, stats = generate_initial_dataset_single_objective(num_samples=20, seed=5)
        recomputed = compute_dataset_statistics(rows)
        self.assertEqual(stats, recomputed)
        self.assertEqual(stats["num_samples"], 20)
        self.assertIn("fraction_ranges", stats)
        self.assertIn("metric_statistics", stats)


if __name__ == "__main__":
    unittest.main()
