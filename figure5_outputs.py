"""Figure 5 输出布局、checkpoint I/O、summary 读取和日志配置。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from figure5_experiment_config import Figure5ExperimentConfig
    from figure5_scalarization import Figure5Setting
else:
    Figure5ExperimentConfig = Any
    Figure5Setting = str


CHECKPOINT_SCHEMA_VERSION = 2
SUMMARY_FILENAME = "figure5_summary.json"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_FIGURE_FILENAME = "figure5.png"
LOG_DIR_NAME = "logs"
RUNNER_LOG_FILENAME = "figure5_runner.log"
PLOT_LOG_FILENAME = "plot_figure5.log"
CHECKPOINT_DIR_NAME = "trajectories"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


@dataclass(frozen=True)
class Figure5OutputLayout:
    """Figure 5 输出目录下的标准路径。"""

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

    def checkpoint_filename(self, setting: Figure5Setting, seed: int) -> str:
        return checkpoint_filename(setting, seed)

    def checkpoint_path(self, setting: Figure5Setting, seed: int) -> Path:
        return self.checkpoint_dir / self.checkpoint_filename(setting, seed)


def figure5_output_layout(output_dir: str | Path) -> Figure5OutputLayout:
    return Figure5OutputLayout(root=Path(output_dir))


def checkpoint_filename(setting: Figure5Setting, seed: int) -> str:
    return f"{setting}_seed_{int(seed)}.json"


def checkpoint_payload(
    *,
    setting: Figure5Setting,
    seed: int,
    state: object,
    config: Figure5ExperimentConfig,
) -> dict[str, Any]:
    """Build a checkpoint tied to one setting/seed/config trajectory."""

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "setting": setting,
        "seed": int(seed),
        "config": config.to_dict(),
        "state": asdict(state),
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write JSON so interruption cannot leave a partial checkpoint."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json_atomic(path, payload)


def load_checkpoint(
    path: str | Path,
    *,
    setting: Figure5Setting,
    seed: int,
    config: Figure5ExperimentConfig,
) -> dict[str, Any]:
    """Load a checkpoint only when all trajectory identity fields match."""

    checkpoint = Path(path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema in {checkpoint}")
    if payload.get("setting") != setting or int(payload.get("seed")) != int(seed):
        raise ValueError(f"Checkpoint metadata does not match requested trajectory: {checkpoint}")
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
    """Configure one Figure 5 file handler without disturbing other handlers."""

    output_path = Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, "_figure5_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(output_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler._figure5_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    return output_path


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_FIGURE_FILENAME",
    "Figure5OutputLayout",
    "MANIFEST_FILENAME",
    "SUMMARY_FILENAME",
    "checkpoint_filename",
    "checkpoint_payload",
    "configure_file_logging_path",
    "figure5_output_layout",
    "load_checkpoint",
    "load_summary",
    "write_checkpoint",
    "write_json_atomic",
]
