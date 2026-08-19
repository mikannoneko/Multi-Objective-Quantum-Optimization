"""Configuration shared by the Figure 4 and Figure 5 experiment runners."""

from __future__ import annotations

from dataclasses import dataclass

from experiment_runtime import require_integer


@dataclass(frozen=True)
class EncodingConfig:
    """Discrete encoding configuration shared by both figures."""

    num_levels: int = 50

    def __post_init__(self) -> None:
        num_levels = require_integer("num_levels", self.num_levels, minimum=2)
        object.__setattr__(self, "num_levels", num_levels)


@dataclass(frozen=True)
class FMConfig:
    """Factorization-machine tuning and device configuration."""

    optuna_trials: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "optuna_trials",
            require_integer("optuna_trials", self.optuna_trials, minimum=0),
        )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class SAConfig:
    """Simulated-annealing reads and sweeps configuration."""

    reads: int = 1000
    sweeps: int = 3000

    def __post_init__(self) -> None:
        object.__setattr__(self, "reads", require_integer("sa_reads", self.reads, minimum=1))
        object.__setattr__(self, "sweeps", require_integer("sa_sweeps", self.sweeps, minimum=1))


__all__ = ["EncodingConfig", "FMConfig", "SAConfig"]
