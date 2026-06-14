from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


AL_MATRIX_VOLUME_FRACTION = 0.8
SECONDARY_PHASE_TOTAL_VOLUME_FRACTION = 0.2
ALPHA_TARGET = 20.0  # ppm/K, paper Sec. 3.1.2
DEFAULT_NUM_SAMPLES = 100
DEFAULT_TOLERANCE = 1e-12
MAX_HILL_ITERATIONS = 10_000


PHASE_ORDER = (
    "eutectic_si",
    "mg2si",
    "al3ni",
    "al2cu",
)


CSV_FIELDNAMES = [
    "sample_id",
    "seed",
    "f1_norm",
    "f2_norm",
    "f3_norm",
    "f4_norm",
    "f1_vol",
    "f2_vol",
    "f3_vol",
    "f4_vol",
    "f_Al_vol",
    "kappa",
    "E",
    "rho",
    "delta_alpha",
    "delta_T",
]


@dataclass(frozen=True)
class PhaseProperties:
    kappa: float
    youngs_modulus: float
    density: float
    alpha: float
    bulk_modulus: float


PHASE_PROPERTIES: Dict[str, PhaseProperties] = {
    "al_matrix": PhaseProperties(kappa=164.0, youngs_modulus=70.0, density=2.7, alpha=23.6, bulk_modulus=70.0),
    "eutectic_si": PhaseProperties(kappa=15.0, youngs_modulus=163.0, density=2.3, alpha=2.6, bulk_modulus=97.0),
    "mg2si": PhaseProperties(kappa=11.0, youngs_modulus=116.0, density=2.0, alpha=7.5, bulk_modulus=59.0),
    "al3ni": PhaseProperties(kappa=104.0, youngs_modulus=140.0, density=4.0, alpha=13.0, bulk_modulus=110.0),
    "al2cu": PhaseProperties(kappa=120.0, youngs_modulus=105.0, density=4.4, alpha=17.0, bulk_modulus=97.0),
}


DELTA_T_M1 = 669.6
DELTA_T_N1 = 85.7
DELTA_T_M2 = 1388.9
DELTA_T_N2 = 177.8
EUTECTIC_SI_VOLUME_FRACTION = 0.128


def _validate_seed(seed: int | None) -> int:
    if seed is None:
        raise ValueError("seed is required")
    return int(seed)


def sample_single_objective_design(rng: random.Random) -> Tuple[float, float, float, float]:
    """
    Supplementary S1.4: generate one feasible normalized composition using
    sorted uniform cut points on the simplex followed by a random permutation.
    """
    cut_points = sorted(rng.uniform(0.0, 1.0) for _ in range(len(PHASE_ORDER) - 1))
    boundaries = [0.0, *cut_points, 1.0]
    fractions = [boundaries[i + 1] - boundaries[i] for i in range(len(PHASE_ORDER))]
    rng.shuffle(fractions)
    return tuple(fractions)  # type: ignore[return-value]


def normalized_to_volume_fractions(normalized_fractions: Sequence[float]) -> Tuple[float, float, float, float]:
    return tuple(SECONDARY_PHASE_TOTAL_VOLUME_FRACTION * fk for fk in normalized_fractions)  # type: ignore[return-value]


def compute_kappa(volume_fractions: Sequence[float]) -> float:
    phase_al = PHASE_PROPERTIES["al_matrix"]
    q_value = 0.0
    for phase_name, fk in zip(PHASE_ORDER, volume_fractions):
        phase = PHASE_PROPERTIES[phase_name]
        q_value += ((phase.kappa - phase_al.kappa) / (2.0 * phase_al.kappa + phase.kappa)) * fk
    return phase_al.kappa * (1.0 + 2.0 * q_value) / (1.0 - q_value)


def compute_youngs_modulus(volume_fractions: Sequence[float], tolerance: float = DEFAULT_TOLERANCE) -> float:
    phase_al = PHASE_PROPERTIES["al_matrix"]
    e_reuss = 1.0 / (
        (AL_MATRIX_VOLUME_FRACTION / phase_al.youngs_modulus)
        + sum(fk / PHASE_PROPERTIES[phase_name].youngs_modulus for phase_name, fk in zip(PHASE_ORDER, volume_fractions))
    )
    e_voigt = (AL_MATRIX_VOLUME_FRACTION * phase_al.youngs_modulus) + sum(
        fk * PHASE_PROPERTIES[phase_name].youngs_modulus for phase_name, fk in zip(PHASE_ORDER, volume_fractions)
    )

    e_hill = 0.5 * (e_reuss + e_voigt)
    s_hill = 0.5 * ((1.0 / e_reuss) + (1.0 / e_voigt))

    for _ in range(MAX_HILL_ITERATIONS):
        if abs(e_hill - (1.0 / s_hill)) <= tolerance * max(1.0, abs(e_hill)):
            return e_hill
        e_hill_next = 0.5 * (e_hill + (1.0 / s_hill))
        s_hill_next = 0.5 * ((1.0 / e_hill) + s_hill)
        e_hill = e_hill_next
        s_hill = s_hill_next

    raise RuntimeError("Hill iteration did not converge")


def compute_density(volume_fractions: Sequence[float]) -> float:
    phase_al = PHASE_PROPERTIES["al_matrix"]
    return (AL_MATRIX_VOLUME_FRACTION * phase_al.density) + sum(
        fk * PHASE_PROPERTIES[phase_name].density for phase_name, fk in zip(PHASE_ORDER, volume_fractions)
    )


def compute_alpha(volume_fractions: Sequence[float]) -> float:
    phase_al = PHASE_PROPERTIES["al_matrix"]
    numerator = AL_MATRIX_VOLUME_FRACTION * phase_al.alpha * phase_al.bulk_modulus
    denominator = AL_MATRIX_VOLUME_FRACTION * phase_al.bulk_modulus
    for phase_name, fk in zip(PHASE_ORDER, volume_fractions):
        phase = PHASE_PROPERTIES[phase_name]
        numerator += fk * phase.alpha * phase.bulk_modulus
        denominator += fk * phase.bulk_modulus
    return numerator / denominator


def compute_delta_t(si_fraction_volume: float) -> float:
    if abs(si_fraction_volume - EUTECTIC_SI_VOLUME_FRACTION) <= DEFAULT_TOLERANCE:
        return 0.0
    if si_fraction_volume < EUTECTIC_SI_VOLUME_FRACTION:
        return (-DELTA_T_M1 * si_fraction_volume) + DELTA_T_N1
    return (DELTA_T_M2 * si_fraction_volume) - DELTA_T_N2


def compute_properties(volume_fractions: Sequence[float]) -> Dict[str, float]:
    alpha = compute_alpha(volume_fractions)
    return {
        "kappa": compute_kappa(volume_fractions),
        "E": compute_youngs_modulus(volume_fractions),
        "rho": compute_density(volume_fractions),
        "delta_alpha": abs(alpha - ALPHA_TARGET),
        "delta_T": compute_delta_t(volume_fractions[0]),
    }


def compute_properties_from_normalized_composition(normalized_fractions: Sequence[float]) -> Dict[str, float]:
    volume_fractions = normalized_to_volume_fractions(normalized_fractions)
    return compute_properties(volume_fractions)


def build_dataset_row(sample_id: int, seed: int, normalized_fractions: Sequence[float]) -> Dict[str, float | int]:
    if len(normalized_fractions) != 4:
        raise ValueError("Expected four normalized fractions")
    if any(fk < 0.0 for fk in normalized_fractions):
        raise ValueError("Normalized fractions must be non-negative")
    total = sum(normalized_fractions)
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"Normalized fractions must sum to 1, got {total}")

    volume_fractions = normalized_to_volume_fractions(normalized_fractions)
    properties = compute_properties(volume_fractions)

    return {
        "sample_id": sample_id,
        "seed": seed,
        "f1_norm": normalized_fractions[0],
        "f2_norm": normalized_fractions[1],
        "f3_norm": normalized_fractions[2],
        "f4_norm": normalized_fractions[3],
        "f1_vol": volume_fractions[0],
        "f2_vol": volume_fractions[1],
        "f3_vol": volume_fractions[2],
        "f4_vol": volume_fractions[3],
        "f_Al_vol": AL_MATRIX_VOLUME_FRACTION,
        **properties,
    }


def compute_dataset_statistics(rows: Sequence[Dict[str, float | int]]) -> Dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty")

    norm_fields = ("f1_norm", "f2_norm", "f3_norm", "f4_norm")
    metric_fields = ("kappa", "E", "rho", "delta_alpha", "delta_T")

    max_sum_deviation = max(abs(sum(float(row[field]) for field in norm_fields) - 1.0) for row in rows)
    fraction_ranges = {
        field: {
            "min": min(float(row[field]) for row in rows),
            "max": max(float(row[field]) for row in rows),
        }
        for field in norm_fields
    }
    metric_statistics = {
        field: {
            "mean": statistics.fmean(float(row[field]) for row in rows),
            "variance": statistics.pvariance(float(row[field]) for row in rows),
        }
        for field in metric_fields
    }

    return {
        "num_samples": len(rows),
        "max_sum_deviation": max_sum_deviation,
        "fraction_ranges": fraction_ranges,
        "metric_statistics": metric_statistics,
    }


def write_dataset_csv(rows: Sequence[Dict[str, float | int]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: f"{float(row[field]):.12f}" if isinstance(row[field], float) else row[field]
                    for field in CSV_FIELDNAMES
                }
            )
    return output


def generate_initial_dataset_single_objective(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int | None = None,
    output_path: str | Path | None = None,
) -> Tuple[List[Dict[str, float | int]], Dict[str, object]]:
    seed = _validate_seed(seed)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rng = random.Random(seed)
    rows = [
        build_dataset_row(index, seed, sample_single_objective_design(rng))
        for index in range(num_samples)
    ]
    stats = compute_dataset_statistics(rows)

    if output_path is not None:
        write_dataset_csv(rows, output_path)

    return rows, stats


def generate_initial_dataset_batch(
    seed_list: Sequence[int],
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> Dict[int, Tuple[List[Dict[str, float | int]], Dict[str, object]]]:
    datasets: Dict[int, Tuple[List[Dict[str, float | int]], Dict[str, object]]] = {}
    for seed in seed_list:
        datasets[int(seed)] = generate_initial_dataset_single_objective(num_samples=num_samples, seed=int(seed))
    return datasets


def _default_output_path(seed: int) -> Path:
    return Path(f"initial_dataset_single_objective_seed_{seed}.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the paper's single-objective initial dataset.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed used for deterministic generation.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of samples to generate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to initial_dataset_single_objective_seed_<seed>.csv",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional path to write dataset statistics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output or _default_output_path(args.seed)
    _, stats = generate_initial_dataset_single_objective(
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=output_path,
    )
    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_path), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
