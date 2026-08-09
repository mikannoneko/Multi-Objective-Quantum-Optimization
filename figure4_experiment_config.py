"""Figure 4 实验配置和单目标优化方向定义。

该模块只保存与运行规模、训练参数、SA 参数和 objective 方向有关的配置。
具体的编码策略在 `figure4_setting_strategies.py`，具体运行循环在 `figure4_pipeline.py`。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
from typing import Any, Literal


RunScale = Literal["paper", "quick", "test"]
SUPPORTED_PRESETS: tuple[RunScale, ...] = ("paper", "quick", "test")


@dataclass(frozen=True)
class ObjectiveSpec:
    """描述一个 Figure 4 objective 的优化方向。

    FM/QUBO 训练统一按“越小越好”处理，因此最大化目标会在训练前取负号；
    best-so-far 曲线仍按真实 objective 方向更新。
    """

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


def _require_positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_non_negative(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True)
class EncodingConfig:
    """离散编码配置；`num_levels` 决定每个 block 可表示的正分数等级数。"""

    num_levels: int = 50

    def __post_init__(self) -> None:
        num_levels = _require_positive("num_levels", self.num_levels)
        if num_levels <= 1:
            raise ValueError("num_levels must be greater than 1")
        object.__setattr__(self, "num_levels", num_levels)


@dataclass(frozen=True)
class FMConfig:
    """FM 训练配置；`optuna_trials=0` 时使用固定超参数，适合快速测试。"""

    optuna_trials: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "optuna_trials",
            _require_non_negative("optuna_trials", self.optuna_trials),
        )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class SAConfig:
    """Simulated annealing 求解配置，对应 D-Wave Ocean `neal` 的 reads 和 sweeps。"""

    reads: int = 1000
    sweeps: int = 3000

    def __post_init__(self) -> None:
        object.__setattr__(self, "reads", _require_positive("sa_reads", self.reads))
        object.__setattr__(self, "sweeps", _require_positive("sa_sweeps", self.sweeps))


@dataclass(frozen=True)
class ExperimentConfig:
    """Figure 4 一次实验运行的完整配置。"""

    num_samples: int = 100
    iterations: int = 600
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    sa: SAConfig = field(default_factory=SAConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_samples", _require_positive("num_samples", self.num_samples))
        object.__setattr__(self, "iterations", _require_positive("iterations", self.iterations))

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
    def sa_reads(self) -> int:
        return self.sa.reads

    @property
    def sa_sweeps(self) -> int:
        return self.sa.sweeps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_config(preset: RunScale, device: str = "cpu") -> ExperimentConfig:
    """把论文规模、快速规模和单元测试规模收敛到同一个配置 dataclass。"""

    if preset == "paper":
        return ExperimentConfig(fm=FMConfig(device=device))
    if preset == "quick":
        return ExperimentConfig(
            num_samples=100,
            iterations=100,
            encoding=EncodingConfig(num_levels=50),
            fm=FMConfig(optuna_trials=3, device=device),
            sa=SAConfig(reads=100, sweeps=500),
        )
    if preset == "test":
        return ExperimentConfig(
            num_samples=10,
            iterations=2,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0, device=device),
            sa=SAConfig(reads=32, sweeps=12),
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
    sa_reads: int | None = None,
    sa_sweeps: int | None = None,
) -> ExperimentConfig:
    """先解析 preset，再应用 CLI override，得到 runner 实际执行的配置。"""

    config = preset_config(preset, device=device)
    if num_samples is not None or iterations is not None:
        config = replace(
            config,
            num_samples=config.num_samples if num_samples is None else num_samples,
            iterations=config.iterations if iterations is None else iterations,
        )
    if num_levels is not None:
        config = replace(config, encoding=EncodingConfig(num_levels=num_levels))
    if optuna_trials is not None:
        config = replace(config, fm=replace(config.fm, optuna_trials=optuna_trials))
    if sa_reads is not None or sa_sweeps is not None:
        config = replace(
            config,
            sa=SAConfig(
                reads=config.sa.reads if sa_reads is None else sa_reads,
                sweeps=config.sa.sweeps if sa_sweeps is None else sa_sweeps,
            ),
        )
    return config
