"""Figure 5 多目标实验规模和训练/求解配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from figure4_experiment_config import EncodingConfig, FMConfig, SAConfig, _require_positive


Figure5RunScale = Literal["paper", "quick", "test"]
SUPPORTED_PRESETS: tuple[Figure5RunScale, ...] = ("paper", "quick", "test")


@dataclass(frozen=True)
class Figure5ExperimentConfig:
    """Figure 5 一次实验的完整配置。

    Figure 5 复用 Figure 4 的 FM、SA 和离散编码参数类，但使用
    独立的 preset，避免把 Figure 4 的 600 轮单目标默认值带入多目标实验。
    """

    num_samples: int = 500
    iterations: int = 1000
    encoding: EncodingConfig = field(default_factory=lambda: EncodingConfig(num_levels=25))
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


def preset_config(preset: Figure5RunScale, device: str = "cpu") -> Figure5ExperimentConfig:
    """Resolve the paper reference, project reproduction, or test scale."""

    if preset == "paper":
        return Figure5ExperimentConfig(fm=FMConfig(device=device))
    if preset == "quick":
        return Figure5ExperimentConfig(
            num_samples=500,
            iterations=150,
            encoding=EncodingConfig(num_levels=25),
            fm=FMConfig(optuna_trials=3, device=device),
            sa=SAConfig(reads=100, sweeps=500),
        )
    if preset == "test":
        return Figure5ExperimentConfig(
            num_samples=10,
            iterations=2,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0, device=device),
            sa=SAConfig(reads=32, sweeps=12),
        )
    raise ValueError(f"Unsupported preset {preset!r}. Choices: {', '.join(SUPPORTED_PRESETS)}")


def resolve_experiment_config(
    *,
    preset: Figure5RunScale,
    device: str,
    num_samples: int | None = None,
    iterations: int | None = None,
    num_levels: int | None = None,
    optuna_trials: int | None = None,
    sa_reads: int | None = None,
    sa_sweeps: int | None = None,
) -> Figure5ExperimentConfig:
    """Resolve a preset first, then apply explicit CLI overrides."""

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


__all__ = [
    "Figure5ExperimentConfig",
    "Figure5RunScale",
    "SUPPORTED_PRESETS",
    "preset_config",
    "resolve_experiment_config",
]
