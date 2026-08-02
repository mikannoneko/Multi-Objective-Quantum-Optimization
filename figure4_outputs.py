"""Figure 4 输出布局、checkpoint I/O、summary 读取和日志文件配置。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, Protocol

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
_CHECKPOINT_FIELDS = frozenset({"schema_version", "objective", "setting", "seed", "config", "state"})


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


def _same_json_value(actual: Any, expected: Any) -> bool:
    """比较 JSON 值，同时保留 bool/int 等类型差异。"""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return bool(actual == expected)


def _checkpoint_error(checkpoint: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid checkpoint {checkpoint}: {detail}")


def load_checkpoint(
    path: str | Path,
    *,
    objective: ObjectiveLike,
    seed: int,
    config: ExperimentConfig,
    setting: Figure4Setting,
) -> dict[str, Any]:
    """读取 checkpoint，并校验外层 schema、类型和 trajectory 身份。"""

    checkpoint = Path(path)
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _checkpoint_error(checkpoint, f"cannot read valid JSON ({exc})") from exc

    if not isinstance(payload, dict):
        raise _checkpoint_error(checkpoint, "root must be a JSON object")

    missing_fields = sorted(_CHECKPOINT_FIELDS - payload.keys())
    if missing_fields:
        raise _checkpoint_error(checkpoint, f"missing top-level fields: {', '.join(missing_fields)}")

    schema_version = payload["schema_version"]
    if type(schema_version) is not int:
        raise _checkpoint_error(checkpoint, "field 'schema_version' must be an integer")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_error(
            checkpoint,
            f"unsupported schema_version {schema_version!r}; expected {CHECKPOINT_SCHEMA_VERSION}",
        )

    checkpoint_objective = payload["objective"]
    checkpoint_setting = payload["setting"]
    checkpoint_seed = payload["seed"]
    if type(checkpoint_objective) is not str:
        raise _checkpoint_error(checkpoint, "field 'objective' must be a string")
    if type(checkpoint_setting) is not str:
        raise _checkpoint_error(checkpoint, "field 'setting' must be a string")
    if type(checkpoint_seed) is not int:
        raise _checkpoint_error(checkpoint, "field 'seed' must be an integer")
    if checkpoint_objective != objective.name or checkpoint_seed != int(seed):
        raise _checkpoint_error(checkpoint, "objective/seed metadata does not match the requested trajectory")
    if checkpoint_setting != setting:
        raise _checkpoint_error(checkpoint, f"setting metadata does not match {setting!r}")

    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, dict):
        raise _checkpoint_error(checkpoint, "field 'config' must be a JSON object")
    if not _same_json_value(checkpoint_config, config.to_dict()):
        raise _checkpoint_error(
            checkpoint,
            "config does not match the current run; use a separate output directory for different settings",
        )
    if not isinstance(payload["state"], dict):
        raise _checkpoint_error(checkpoint, "field 'state' must be a JSON object")
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
