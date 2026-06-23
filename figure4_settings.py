from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import numpy as np

from figure4_qubo_math import (
    IterationEncoding,
    create_cgfm_iteration_encoding,
    create_iteration_encoding,
    decode_candidate_bits_to_cgfm_composition,
    decode_candidate_bits_to_composition,
    encode_cgfm_rows,
    encode_single_objective_rows,
)


Figure4Setting = Literal["wo_cgfm", "w_cgfm"]
SUPPORTED_SETTINGS: tuple[Figure4Setting, ...] = ("wo_cgfm", "w_cgfm")


class SettingStrategy(Protocol):
    name: Figure4Setting
    include_system_penalty: bool

    def create_encoding(self, num_levels: int, seed: int) -> IterationEncoding:
        ...

    def encode_rows(self, rows: Sequence[dict[str, Any]], encoding: IterationEncoding) -> np.ndarray:
        ...

    def decode_candidate(self, candidate_bits: np.ndarray, encoding: IterationEncoding) -> np.ndarray | None:
        ...


@dataclass(frozen=True)
class WOCGFMStrategy:
    name: Figure4Setting = "wo_cgfm"
    include_system_penalty: bool = True

    def create_encoding(self, num_levels: int, seed: int) -> IterationEncoding:
        return create_iteration_encoding(num_levels, seed)

    def encode_rows(self, rows: Sequence[dict[str, Any]], encoding: IterationEncoding) -> np.ndarray:
        return encode_single_objective_rows(rows, encoding)

    def decode_candidate(self, candidate_bits: np.ndarray, encoding: IterationEncoding) -> np.ndarray | None:
        return decode_candidate_bits_to_composition(candidate_bits, encoding)


@dataclass(frozen=True)
class WCGFMStrategy:
    name: Figure4Setting = "w_cgfm"
    include_system_penalty: bool = False

    def create_encoding(self, num_levels: int, seed: int) -> IterationEncoding:
        return create_cgfm_iteration_encoding(num_levels, seed)

    def encode_rows(self, rows: Sequence[dict[str, Any]], encoding: IterationEncoding) -> np.ndarray:
        return encode_cgfm_rows(rows, encoding)

    def decode_candidate(self, candidate_bits: np.ndarray, encoding: IterationEncoding) -> np.ndarray | None:
        return decode_candidate_bits_to_cgfm_composition(candidate_bits, encoding)


SETTING_STRATEGIES: dict[Figure4Setting, SettingStrategy] = {
    "wo_cgfm": WOCGFMStrategy(),
    "w_cgfm": WCGFMStrategy(),
}


def get_setting_strategy(setting: str) -> SettingStrategy:
    if setting not in SETTING_STRATEGIES:
        raise ValueError(f"Unsupported setting {setting!r}. Choices: {', '.join(SUPPORTED_SETTINGS)}")
    return SETTING_STRATEGIES[setting]  # type: ignore[index]


def validate_settings(settings: Sequence[str]) -> tuple[Figure4Setting, ...]:
    if not settings:
        raise ValueError("At least one setting is required.")
    return tuple(get_setting_strategy(setting).name for setting in settings)
