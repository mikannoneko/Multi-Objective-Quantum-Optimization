from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import dimod
import numpy as np
from neal import SimulatedAnnealingSampler


SYSTEM_PENALTY_WEIGHT = 650.0
ONE_HOT_PENALTY_WEIGHT = 1.0
DEFAULT_TOLERANCE = 1e-10
CGFM_ANGLE_MAX = np.pi / 2.0


@dataclass(frozen=True)
class IterationEncoding:
    num_levels: int
    positive_count_by_bit: Tuple[np.ndarray, ...]
    bit_index_by_count: Tuple[np.ndarray, ...]
    value_scale: float = 1.0
    phase_permutation: Optional[np.ndarray] = None

    @property
    def num_blocks(self) -> int:
        return len(self.positive_count_by_bit)


def get_positive_level_values(num_levels: int) -> np.ndarray:
    if num_levels <= 1:
        raise ValueError("num_levels must be greater than 1")
    return np.arange(1, num_levels + 1, dtype=np.int64)


def create_iteration_encoding(
    num_levels: int,
    seed: int,
    num_blocks: int = 4,
    value_scale: float = 1.0,
    phase_permutation: Optional[np.ndarray] = None,
) -> IterationEncoding:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    rng = np.random.default_rng(seed)
    counts = get_positive_level_values(num_levels)
    positive_count_by_bit = []
    bit_index_by_count = []
    for _ in range(num_blocks):
        shuffled = rng.permutation(counts)
        reverse = np.full(num_levels + 1, -1, dtype=np.int64)
        for bit_idx, count_value in enumerate(shuffled):
            reverse[int(count_value)] = int(bit_idx)
        positive_count_by_bit.append(shuffled.astype(np.int64))
        bit_index_by_count.append(reverse)
    return IterationEncoding(
        num_levels=num_levels,
        positive_count_by_bit=tuple(positive_count_by_bit),
        bit_index_by_count=tuple(bit_index_by_count),
        value_scale=float(value_scale),
        phase_permutation=None if phase_permutation is None else np.asarray(phase_permutation, dtype=np.int64),
    )


def create_cgfm_iteration_encoding(num_levels: int, seed: int) -> IterationEncoding:
    rng = np.random.default_rng(seed)
    phase_permutation = rng.permutation(4)
    return create_iteration_encoding(
        num_levels=num_levels,
        seed=seed,
        num_blocks=3,
        value_scale=CGFM_ANGLE_MAX,
        phase_permutation=phase_permutation,
    )


def evaluate_qubo_energy(bit_vector: np.ndarray, q: np.ndarray, bias: float = 0.0) -> float:
    bits = np.asarray(bit_vector, dtype=np.float64)
    return float(bits @ q @ bits + bias)


def validate_candidate_composition(composition: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    return bool(np.all(composition >= -tolerance) and abs(float(np.sum(composition)) - 1.0) <= tolerance)


def _quantize_composition_to_counts(composition: np.ndarray, num_levels: int) -> np.ndarray:
    scaled = np.floor((np.asarray(composition, dtype=np.float64) * num_levels) + 0.5)
    return np.clip(scaled, 0, num_levels).astype(np.int64)


def _discrete_error(count: int, target: float, num_levels: int) -> float:
    return abs((count / num_levels) - target)


def cure_discrete_counts(composition: Sequence[float], num_levels: int) -> np.ndarray:
    composition_array = np.asarray(composition, dtype=np.float64)
    counts = _quantize_composition_to_counts(composition_array, num_levels)
    deficit = int(num_levels - int(np.sum(counts)))

    while deficit != 0:
        if deficit > 0:
            candidates = [
                (
                    _discrete_error(int(counts[idx]) + 1, float(composition_array[idx]), num_levels)
                    - _discrete_error(int(counts[idx]), float(composition_array[idx]), num_levels),
                    idx,
                )
                for idx in range(4)
                if counts[idx] < num_levels
            ]
            if not candidates:
                break
            _, chosen_idx = min(candidates)
            counts[chosen_idx] += 1
            deficit -= 1
        else:
            candidates = [
                (
                    _discrete_error(int(counts[idx]) - 1, float(composition_array[idx]), num_levels)
                    - _discrete_error(int(counts[idx]), float(composition_array[idx]), num_levels),
                    idx,
                )
                for idx in range(4)
                if counts[idx] > 0
            ]
            if not candidates:
                break
            _, chosen_idx = min(candidates)
            counts[chosen_idx] -= 1
            deficit += 1

    if int(np.sum(counts)) != num_levels:
        raise RuntimeError("bit-SWAP curing failed to satisfy the system constraint")
    return counts


def counts_to_composition(counts: Sequence[int], num_levels: int) -> np.ndarray:
    return np.asarray(counts, dtype=np.float64) / float(num_levels)


def prepare_discrete_composition(composition: Sequence[float], num_levels: int) -> np.ndarray:
    return counts_to_composition(cure_discrete_counts(composition, num_levels), num_levels)


def encode_discrete_values(values: Sequence[float], encoding: IterationEncoding) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.shape[0] != encoding.num_blocks:
        raise ValueError(f"Expected {encoding.num_blocks} values, got {value_array.shape[0]}")
    counts = np.rint((value_array / encoding.value_scale) * encoding.num_levels).astype(np.int64)
    counts = np.clip(counts, 0, encoding.num_levels)
    bit_vector = np.zeros(encoding.num_blocks * encoding.num_levels, dtype=np.float32)
    for block_idx, count_value in enumerate(counts):
        if count_value <= 0:
            continue
        bit_idx = int(encoding.bit_index_by_count[block_idx][int(count_value)])
        if bit_idx < 0:
            raise RuntimeError("Missing bit assignment for discrete count")
        bit_vector[(block_idx * encoding.num_levels) + bit_idx] = 1.0
    return bit_vector


def encode_discrete_composition(composition: Sequence[float], encoding: IterationEncoding) -> np.ndarray:
    if encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE:
        raise ValueError("Direct composition encoding requires four unit-scale blocks")
    return encode_discrete_values(composition, encoding)


def cgfm_composition_to_angles(
    composition: Sequence[float],
    phase_permutation: Sequence[int],
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    composition_array = np.asarray(composition, dtype=np.float64)
    permutation = np.asarray(phase_permutation, dtype=np.int64)
    if composition_array.shape[0] != 4 or permutation.shape[0] != 4:
        raise ValueError("CGFM requires four phase fractions and a four-element permutation")
    permuted = np.clip(composition_array[permutation], 0.0, None)
    total = float(np.sum(permuted))
    if total <= tolerance:
        raise ValueError("Composition total must be positive")
    permuted = permuted / total

    angles = np.zeros(3, dtype=np.float64)
    for angle_idx in range(3):
        current = float(permuted[angle_idx])
        tail = float(np.sum(permuted[angle_idx + 1 :]))
        if current <= tolerance and tail <= tolerance:
            angles[angle_idx] = 0.0
        else:
            angles[angle_idx] = np.arctan2(np.sqrt(max(tail, 0.0)), np.sqrt(max(current, 0.0)))
    return np.clip(angles, 0.0, CGFM_ANGLE_MAX)


def cgfm_angles_to_composition(
    angles: Sequence[float],
    phase_permutation: Sequence[int],
) -> np.ndarray:
    angle_array = np.clip(np.asarray(angles, dtype=np.float64), 0.0, CGFM_ANGLE_MAX)
    permutation = np.asarray(phase_permutation, dtype=np.int64)
    if angle_array.shape[0] != 3 or permutation.shape[0] != 4:
        raise ValueError("CGFM requires three angles and a four-element permutation")

    s1, s2, s3 = np.sin(angle_array)
    c1, c2, c3 = np.cos(angle_array)
    permuted = np.array(
        [
            c1 * c1,
            (s1 * s1) * (c2 * c2),
            (s1 * s1) * (s2 * s2) * (c3 * c3),
            (s1 * s1) * (s2 * s2) * (s3 * s3),
        ],
        dtype=np.float64,
    )
    total = float(np.sum(permuted))
    if total > 0.0:
        permuted /= total
    composition = np.zeros(4, dtype=np.float64)
    composition[permutation] = permuted
    return composition


def encode_single_objective_rows(
    rows: Sequence[Dict[str, float | int]],
    encoding: IterationEncoding,
) -> np.ndarray:
    features = np.zeros((len(rows), encoding.num_blocks * encoding.num_levels), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        composition = np.array(
            [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
            dtype=np.float64,
        )
        features[row_idx] = encode_discrete_composition(composition, encoding)
    return features


def encode_cgfm_rows(
    rows: Sequence[Dict[str, float | int]],
    encoding: IterationEncoding,
) -> np.ndarray:
    if encoding.num_blocks != 3 or encoding.phase_permutation is None:
        raise ValueError("CGFM encoding requires three blocks and a phase permutation")
    features = np.zeros((len(rows), encoding.num_blocks * encoding.num_levels), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        composition = np.array(
            [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
            dtype=np.float64,
        )
        angles = cgfm_composition_to_angles(composition, encoding.phase_permutation)
        features[row_idx] = encode_discrete_values(angles, encoding)
    return features


def build_system_penalty_matrix(encoding: IterationEncoding) -> Tuple[np.ndarray, float]:
    if encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE:
        raise ValueError("System penalty is defined only for direct four-fraction encoding")
    alpha = np.concatenate(
        [block.astype(np.float64) / float(encoding.num_levels) for block in encoding.positive_count_by_bit]
    )
    q = np.outer(alpha, alpha)
    q[np.diag_indices_from(q)] -= 2.0 * alpha
    return q, 1.0


def build_one_hot_penalty_matrix(encoding: IterationEncoding) -> Tuple[np.ndarray, float]:
    size = encoding.num_blocks * encoding.num_levels
    q = np.zeros((size, size), dtype=np.float64)
    for block_idx in range(encoding.num_blocks):
        start = block_idx * encoding.num_levels
        end = start + encoding.num_levels
        q[start:end, start:end] += 1.0
        q[np.arange(start, end), np.arange(start, end)] = 0.0
    return q, 0.0


def normalize_qubo_term(q: np.ndarray, bias: float) -> Tuple[np.ndarray, float, float]:
    scale = max(float(np.max(np.abs(q))), abs(float(bias)))
    if scale <= 0.0:
        return q.copy(), float(bias), 1.0
    return q / scale, float(bias) / scale, scale


def build_single_objective_qubo(
    fm_q: np.ndarray,
    fm_bias: float,
    encoding: IterationEncoding,
    include_system_penalty: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    one_hot_q, one_hot_bias = build_one_hot_penalty_matrix(encoding)

    normalized_fm_q, normalized_fm_bias, fm_scale = normalize_qubo_term(fm_q, fm_bias)
    normalized_one_hot_q, normalized_one_hot_bias, one_hot_scale = normalize_qubo_term(one_hot_q, one_hot_bias)

    total_q = normalized_fm_q + (ONE_HOT_PENALTY_WEIGHT * normalized_one_hot_q)
    total_bias = normalized_fm_bias + (ONE_HOT_PENALTY_WEIGHT * normalized_one_hot_bias)
    system_scale = 0.0
    system_penalty_weight = 0.0
    if include_system_penalty:
        system_q, system_bias = build_system_penalty_matrix(encoding)
        normalized_system_q, normalized_system_bias, system_scale = normalize_qubo_term(system_q, system_bias)
        total_q = total_q + (SYSTEM_PENALTY_WEIGHT * normalized_system_q)
        total_bias = total_bias + (SYSTEM_PENALTY_WEIGHT * normalized_system_bias)
        system_penalty_weight = SYSTEM_PENALTY_WEIGHT

    stats = {
        "fm_scale": fm_scale,
        "system_scale": system_scale,
        "one_hot_scale": one_hot_scale,
        "system_penalty_weight": system_penalty_weight,
        "one_hot_penalty_weight": ONE_HOT_PENALTY_WEIGHT,
        "num_variables": float(total_q.shape[0]),
        "max_abs": float(np.max(np.abs(total_q))),
    }
    return total_q, total_bias, stats


def decode_candidate_bits_to_values(
    bit_vector: np.ndarray,
    encoding: IterationEncoding,
) -> Optional[np.ndarray]:
    bit_array = np.asarray(bit_vector, dtype=np.float64)
    if bit_array.shape[0] != encoding.num_blocks * encoding.num_levels:
        raise ValueError("Unexpected bit vector length")
    values = np.zeros(encoding.num_blocks, dtype=np.float64)
    for block_idx in range(encoding.num_blocks):
        start = block_idx * encoding.num_levels
        end = start + encoding.num_levels
        block = bit_array[start:end]
        active_indices = np.flatnonzero(block > 0.5)
        if len(active_indices) > 1:
            return None
        if len(active_indices) == 1:
            count_value = int(encoding.positive_count_by_bit[block_idx][int(active_indices[0])])
            values[block_idx] = (count_value / float(encoding.num_levels)) * encoding.value_scale
    return values


def decode_candidate_bits_to_composition(
    bit_vector: np.ndarray,
    encoding: IterationEncoding,
) -> Optional[np.ndarray]:
    values = decode_candidate_bits_to_values(bit_vector, encoding)
    if values is None:
        return None
    if encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE:
        raise ValueError("Direct composition decoding requires four unit-scale blocks")
    return values


def decode_candidate_bits_to_cgfm_composition(
    bit_vector: np.ndarray,
    encoding: IterationEncoding,
) -> Optional[np.ndarray]:
    values = decode_candidate_bits_to_values(bit_vector, encoding)
    if values is None:
        return None
    if encoding.num_blocks != 3 or encoding.phase_permutation is None:
        raise ValueError("CGFM decoding requires three blocks and a phase permutation")
    return cgfm_angles_to_composition(values, encoding.phase_permutation)


def _qubo_to_bqm(q: np.ndarray, bias: float) -> dimod.BinaryQuadraticModel:
    linear = {idx: float(q[idx, idx]) for idx in range(q.shape[0])}
    quadratic = {}
    for i in range(q.shape[0]):
        for j in range(i + 1, q.shape[1]):
            coefficient = float(q[i, j] + q[j, i])
            if abs(coefficient) > 0.0:
                quadratic[(i, j)] = coefficient
    return dimod.BinaryQuadraticModel(linear, quadratic, float(bias), dimod.BINARY)


def simulated_annealing_qubo(
    q: np.ndarray,
    bias: float,
    runs: int,
    sweeps: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    bqm = _qubo_to_bqm(q, bias)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=max(1, runs), num_sweeps=max(1, sweeps), seed=seed)
    best_sample = sampleset.first.sample
    best_state = np.array([best_sample[idx] for idx in range(q.shape[0])], dtype=np.float64)
    best_energy = float(sampleset.first.energy)
    return best_state, best_energy
