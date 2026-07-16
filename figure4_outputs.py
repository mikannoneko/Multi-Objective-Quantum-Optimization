"""Figure 4 输出布局、checkpoint I/O、summary 读取和日志文件配置。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from figure4_experiment_config import ExperimentConfig
    from figure4_setting_strategies import Figure4Setting
else:
    ExperimentConfig = Any
    Figure4Setting = str


CHECKPOINT_SCHEMA_VERSION = 3
SUMMARY_FILENAME = "figure4_summary.json"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_FIGURE_FILENAME = "figure4.png"
LOG_DIR_NAME = "logs"
RUNNER_LOG_FILENAME = "figure4_runner.log"
PLOT_LOG_FILENAME = "plot_figure4.log"
CHECKPOINT_DIR_NAME = "trajectories"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

OutputCategory = Literal["managed_current", "protected_legacy", "temporary_test", "unknown"]

PROTECTED_LEGACY_OUTPUT_NAMES = frozenset(
    {
        "figure4_quick_l50",
        "figure4_cgfm_quick_l50",
        "figure4_compare_l50_from_separate",
    }
)
PROTECTED_LEGACY_NAME_FRAGMENTS = ("quick_150", "quick150")
TEMPORARY_OUTPUT_NAMES = frozenset({".tmp_test", ".pytest_cache", "__pycache__"})


class ObjectiveLike(Protocol):
    name: str


@dataclass(frozen=True)
class Figure4OutputLayout:
    """集中定义一个 Figure 4 输出目录下的所有标准路径。"""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def summary_path(self) -> Path:
        return self.root / SUMMARY_FILENAME

    @property
    def figure_path(self) -> Path:
        return self.root / DEFAULT_FIGURE_FILENAME

    @property
    def logs_dir(self) -> Path:
        return self.root / LOG_DIR_NAME

    @property
    def runner_log_path(self) -> Path:
        return self.logs_dir / RUNNER_LOG_FILENAME

    @property
    def plot_log_path(self) -> Path:
        return self.logs_dir / PLOT_LOG_FILENAME

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / CHECKPOINT_DIR_NAME

    def checkpoint_filename(self, setting: Figure4Setting, objective_name: str, seed: int) -> str:
        return checkpoint_filename(setting, objective_name, seed)

    def checkpoint_path(self, setting: Figure4Setting, objective_name: str, seed: int) -> Path:
        return self.checkpoint_dir / self.checkpoint_filename(setting, objective_name, seed)


def figure4_output_layout(output_dir: str | Path) -> Figure4OutputLayout:
    return Figure4OutputLayout(root=Path(output_dir))


def checkpoint_filename(setting: Figure4Setting, objective_name: str, seed: int) -> str:
    return f"{setting}_{objective_name}_seed_{int(seed)}.json"


def checkpoint_payload(
    *,
    objective: ObjectiveLike,
    setting: Figure4Setting,
    seed: int,
    state: object,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """构造 schema v3 checkpoint payload。

    checkpoint 绑定 objective、setting、seed 和 config；恢复时这些字段必须一致。
    """

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "objective": objective.name,
        "setting": setting,
        "seed": int(seed),
        "config": config.to_dict(),
        "state": asdict(state),
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """原子写入 JSON，避免中断时留下半截文件。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """原子写入 checkpoint。"""

    return write_json_atomic(path, payload)


def load_checkpoint(
    path: str | Path,
    *,
    objective: ObjectiveLike,
    seed: int,
    config: ExperimentConfig,
    setting: Figure4Setting,
) -> dict[str, Any]:
    """读取并校验 checkpoint 是否属于当前请求的 trajectory。"""

    checkpoint = Path(path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema in {checkpoint}")
    if payload.get("objective") != objective.name or int(payload.get("seed")) != int(seed):
        raise ValueError(f"Checkpoint metadata does not match requested trajectory: {checkpoint}")
    if payload.get("setting") != setting:
        raise ValueError(f"Checkpoint setting is not {setting}: {checkpoint}")
    if payload.get("config") != config.to_dict():
        raise ValueError(
            f"Checkpoint config does not match current config: {checkpoint}. "
            "Use a separate output directory for different run settings."
        )
    if "state" not in payload:
        raise ValueError(f"Checkpoint is missing state: {checkpoint}")
    return payload


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configure_file_logging_path(log_path: str | Path) -> Path:
    """为 runner/plot 配置单个文件日志 handler。"""

    output_path = Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, "_figure4_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(output_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler._figure4_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    return output_path


def _normalized_path_parts(path: str | Path) -> list[str]:
    return [part.lower() for part in Path(path).parts]


def _looks_like_figure4_output_name(name: str) -> bool:
    return name.startswith("figure4_") and Path(name).suffix == ""


def classify_output_path(path: str | Path) -> OutputCategory:
    """区分当前受管理输出、历史受保护输出和临时测试输出。"""

    parts = _normalized_path_parts(path)
    if any(part in PROTECTED_LEGACY_OUTPUT_NAMES for part in parts):
        return "protected_legacy"
    if any(fragment in part for part in parts for fragment in PROTECTED_LEGACY_NAME_FRAGMENTS):
        return "protected_legacy"
    if any(part in TEMPORARY_OUTPUT_NAMES for part in parts):
        return "temporary_test"
    if any(part.startswith("smoke_figure4") for part in parts):
        return "temporary_test"
    if any(_looks_like_figure4_output_name(part) and part.endswith("_test") for part in parts):
        return "temporary_test"
    if any(_looks_like_figure4_output_name(part) for part in parts):
        return "managed_current"
    return "unknown"
