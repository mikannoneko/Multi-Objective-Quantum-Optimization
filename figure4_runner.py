"""Figure 4 命令行入口。

Runner 只负责解析 CLI、检查依赖/CUDA、写 manifest 和调用 pipeline；实际 active
learning 逻辑保留在 `figure4_pipeline.py`。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from figure4_experiment_config import OBJECTIVES, SUPPORTED_PRESETS, ObjectiveSpec, RunScale, resolve_experiment_config
from figure4_outputs import Figure4OutputLayout, configure_file_logging_path, figure4_output_layout
from figure4_pipeline import ensure_training_dependencies, run_figure4_experiment
from figure4_setting_strategies import SUPPORTED_SETTINGS


LOGGER = logging.getLogger(__name__)
PRESET_NUM_SEEDS: dict[RunScale, int] = {
    "paper": 20,
    "quick_l50": 3,
    "test": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Figure 4 PyTorch FM reproduction pipeline.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary outputs.")
    parser.add_argument(
        "--preset",
        choices=SUPPORTED_PRESETS,
        default="paper",
        help="Resolved defaults: paper, quick_l50, or test. Explicit numeric options override the preset.",
    )
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of initial datasets / trajectories.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the contiguous seed list.")
    parser.add_argument("--iterations", type=int, default=None, help="Active learning iterations per trajectory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Initial dataset size per seed.")
    parser.add_argument("--num-levels", type=int, default=None, help="One-hot levels per variable block.")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Number of Optuna trials for FM tuning.")
    parser.add_argument("--sa-runs", type=int, default=None, help="Simulated annealing restart count.")
    parser.add_argument("--sa-sweeps", type=int, default=None, help="Simulated annealing sweep count.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch execution device.")
    parser.add_argument("--resume", action="store_true", help="Resume from per-trajectory checkpoints in output-dir.")
    parser.add_argument(
        "--settings",
        nargs="+",
        required=True,
        choices=SUPPORTED_SETTINGS,
        help="Figure 4 settings to run. Choices: wo_cgfm w_cgfm",
    )
    parser.add_argument(
        "--objectives",
        nargs="*",
        default=None,
        help="Optional subset of objectives to run. Choices: kappa E rho delta_alpha delta_T",
    )
    return parser.parse_args()


def _selected_objectives(requested_names: list[str] | None) -> tuple[ObjectiveSpec, ...]:
    """按 canonical `OBJECTIVES` 顺序筛选 CLI 请求的 objective 子集。"""

    if not requested_names:
        return OBJECTIVES
    requested = set(requested_names)
    selected = tuple(objective for objective in OBJECTIVES if objective.name in requested)
    if not selected:
        raise ValueError("No valid objectives selected.")
    unknown = requested - {objective.name for objective in OBJECTIVES}
    if unknown:
        raise ValueError(f"Unknown objectives: {', '.join(sorted(unknown))}")
    return selected


def _write_manifest(
    *,
    output_layout: Figure4OutputLayout,
    args: argparse.Namespace,
    resolved_config: object,
    seed_list: list[int],
    objective_names: list[str],
    settings: list[str],
) -> Path:
    """记录本次运行命令、解释器、torch/cuda 状态和解析后的配置。"""

    manifest = {
        "command": [sys.executable, *sys.argv],
        "preset": args.preset,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "resolved_config": asdict(resolved_config),
        "output_paths": {
            "root": str(output_layout.root),
            "summary": str(output_layout.summary_path),
            "manifest": str(output_layout.manifest_path),
            "runner_log": str(output_layout.runner_log_path),
            "plot_log": str(output_layout.plot_log_path),
            "checkpoint_dir": str(output_layout.checkpoint_dir),
        },
        "resume": bool(args.resume),
        "seed_list": seed_list,
        "objectives": objective_names,
        "settings": settings,
    }
    output_layout.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_layout.manifest_path


def _ensure_requested_device_available(device: str) -> None:
    """用户显式请求 CUDA 时，训练开始前尽早失败。"""

    if device == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("Requested --device cuda, but torch.cuda.is_available() is false.")


def main() -> None:
    """解析命令行并启动 Figure 4 实验。"""

    args = parse_args()
    output_layout = figure4_output_layout(args.output_dir)
    log_path = configure_file_logging_path(output_layout.runner_log_path)
    LOGGER.info("Starting Figure 4 runner")
    LOGGER.info("Command: %s", " ".join(sys.argv))
    ensure_training_dependencies()
    selected_objectives = _selected_objectives(args.objectives)
    config = resolve_experiment_config(
        preset=args.preset,
        device=args.device,
        num_samples=args.num_samples,
        iterations=args.iterations,
        num_levels=args.num_levels,
        optuna_trials=args.optuna_trials,
        sa_runs=args.sa_runs,
        sa_sweeps=args.sa_sweeps,
    )
    _ensure_requested_device_available(config.device)
    num_seeds = PRESET_NUM_SEEDS[args.preset] if args.num_seeds is None else int(args.num_seeds)
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    seed_list = list(range(args.seed_start, args.seed_start + num_seeds))
    LOGGER.info("Resolved config: %s", json.dumps(asdict(config), sort_keys=True))
    LOGGER.info(
        "Run selection: preset=%s seeds=%s objectives=%s settings=%s resume=%s",
        args.preset,
        seed_list,
        [obj.name for obj in selected_objectives],
        list(args.settings),
        bool(args.resume),
    )

    summary = run_figure4_experiment(
        seed_list=seed_list,
        config=config,
        output_dir=args.output_dir,
        objectives=selected_objectives,
        resume=args.resume,
        settings=args.settings,
    )
    manifest_path = _write_manifest(
        output_layout=output_layout,
        args=args,
        resolved_config=config,
        seed_list=seed_list,
        objective_names=[obj.name for obj in selected_objectives],
        settings=list(args.settings),
    )
    LOGGER.info("Finished Figure 4 runner; summary=%s manifest=%s", output_layout.summary_path, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "preset": args.preset,
                "objectives": [obj.name for obj in selected_objectives],
                "settings": list(args.settings),
                "training_backend": summary.training_backend,
                "manifest": str(manifest_path),
                "log": str(log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
