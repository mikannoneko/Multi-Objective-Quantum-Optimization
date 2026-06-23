from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal


RunScale = Literal["paper", "quick_l50", "test"]
SUPPORTED_PRESETS: tuple[RunScale, ...] = ("paper", "quick_l50", "test")


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    maximize: bool

    def transform_for_training(self, values: Any) -> Any:
        return -values if self.maximize else values

    def initial_best(self) -> float:
        return -math.inf if self.maximize else math.inf

    def update_best(self, current_best: float, candidate: float) -> float:
        if self.maximize:
            return max(current_best, candidate)
        return min(current_best, candidate)


OBJECTIVES: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("kappa", maximize=True),
    ObjectiveSpec("E", maximize=True),
    ObjectiveSpec("rho", maximize=False),
    ObjectiveSpec("delta_alpha", maximize=False),
    ObjectiveSpec("delta_T", maximize=False),
)


def _require_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EncodingConfig:
    num_levels: int = 50

    def __post_init__(self) -> None:
        if int(self.num_levels) <= 1:
            raise ValueError("num_levels must be greater than 1")


@dataclass(frozen=True)
class FMConfig:
    optuna_trials: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        _require_non_negative("optuna_trials", self.optuna_trials)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class SAConfig:
    runs: int = 1000
    sweeps: int = 3000

    def __post_init__(self) -> None:
        _require_positive("sa_runs", self.runs)
        _require_positive("sa_sweeps", self.sweeps)


@dataclass(frozen=True)
class ExperimentConfig:
    num_samples: int = 100
    iterations: int = 600
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    sa: SAConfig = field(default_factory=SAConfig)

    def __post_init__(self) -> None:
        _require_positive("num_samples", self.num_samples)
        _require_positive("iterations", self.iterations)

    @property
    def num_levels(self) -> int:
        return self.encoding.num_levels

    @property
    def optuna_trials(self) -> int:
        return self.fm.optuna_trials

    @property
    def device(self) -> str:
        return self.fm.device

    @property
    def sa_runs(self) -> int:
        return self.sa.runs

    @property
    def sa_sweeps(self) -> int:
        return self.sa.sweeps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_config(preset: RunScale, device: str = "cpu") -> ExperimentConfig:
    if preset == "paper":
        return ExperimentConfig(fm=FMConfig(device=device))
    if preset == "quick_l50":
        return ExperimentConfig(
            num_samples=100,
            iterations=100,
            encoding=EncodingConfig(num_levels=50),
            fm=FMConfig(optuna_trials=3, device=device),
            sa=SAConfig(runs=100, sweeps=500),
        )
    if preset == "test":
        return ExperimentConfig(
            num_samples=10,
            iterations=2,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0, device=device),
            sa=SAConfig(runs=2, sweeps=6),
        )
    raise ValueError(f"Unsupported preset {preset!r}. Choices: {', '.join(SUPPORTED_PRESETS)}")


def resolve_experiment_config(
    *,
    preset: RunScale,
    device: str,
    num_samples: int | None = None,
    iterations: int | None = None,
    num_levels: int | None = None,
    optuna_trials: int | None = None,
    sa_runs: int | None = None,
    sa_sweeps: int | None = None,
) -> ExperimentConfig:
    config = preset_config(preset, device=device)
    if num_samples is not None or iterations is not None:
        config = replace(
            config,
            num_samples=config.num_samples if num_samples is None else int(num_samples),
            iterations=config.iterations if iterations is None else int(iterations),
        )
    if num_levels is not None:
        config = replace(config, encoding=EncodingConfig(num_levels=int(num_levels)))
    if optuna_trials is not None:
        config = replace(config, fm=replace(config.fm, optuna_trials=int(optuna_trials)))
    if sa_runs is not None or sa_sweeps is not None:
        config = replace(
            config,
            sa=SAConfig(
                runs=config.sa.runs if sa_runs is None else int(sa_runs),
                sweeps=config.sa.sweeps if sa_sweeps is None else int(sa_sweeps),
            ),
        )
    return config
