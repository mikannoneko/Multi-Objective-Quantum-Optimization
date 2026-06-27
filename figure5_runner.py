"""Figure 5 命令行入口。

Runner 只负责解析 CLI、检查训练环境、写 manifest 并调用 pipeline。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import torch

from figure5_experiment_config import (
    SUPPORTED_PRESETS,
    Figure5RunScale,
    resolve_experiment_config,
)
from figure5_outputs import Figure5OutputLayout, configure_file_logging_path, figure5_output_layout, write_json_atomic
from figure5_pipeline import SUPPORTED_SETTINGS, ensure_training_dependencies, run_figure5_experiment


LOGGER = logging.getLogger(__name__)
PRESET_NUM_SEEDS: dict[Figure5RunScale, int] = {
    "paper": 20,
    "quick150": 3,
    "test": 1,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Figure 5 multi-objective FM+QO reproduction pipeline.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Figure 5 outputs.")
    parser.add_argument(
        "--preset",
        choices=SUPPORTED_PRESETS,
        default="quick150",
        help="Run scale. Defaults to the project reproduction scale quick150.",
    )
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of setting/seed trajectories.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the contiguous seed list.")
    parser.add_argument("--iterations", type=int, default=None, help="Active-learning iterations per trajectory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Initial multi-objective dataset size.")
    parser.add_argument("--num-levels", type=int, default=None, help="Direct one-hot levels per phase block.")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Optuna trials per FM training.")
    parser.add_argument("--sa-runs", type=int, default=None, help="Simulated annealing read count.")
    parser.add_argument("--sa-sweeps", type=int, default=None, help="Simulated annealing sweep count.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch execution device.")
    parser.add_argument("--resume", action="store_true", help="Resume matching trajectory checkpoints.")
    parser.add_argument(
        "--settings",
        nargs="+",
        required=True,
        choices=SUPPORTED_SETTINGS,
        help="Figure 5 comparison settings: w_ddts wo_ddts",
    )
    return parser.parse_args(argv)


def _ensure_requested_device_available(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("Requested --device cuda, but torch.cuda.is_available() is false.")


def _write_manifest(
    *,
    output_layout: Figure5OutputLayout,
    args: argparse.Namespace,
    resolved_config: object,
    seed_list: list[int],
    settings: list[str],
) -> Path:
    config_dict = resolved_config.to_dict()  # type: ignore[attr-defined]
    manifest = {
        "command": [sys.executable, *sys.argv],
        "preset": args.preset,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "resolved_config": config_dict,
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
        "objectives": ["kappa", "E", "rho"],
        "settings": settings,
    }
    return write_json_atomic(output_layout.manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_layout = figure5_output_layout(args.output_dir)
    log_path = configure_file_logging_path(output_layout.runner_log_path)
    LOGGER.info("Starting Figure 5 runner")
    ensure_training_dependencies()
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
    LOGGER.info(
        "Resolved Figure 5 config=%s seeds=%s settings=%s resume=%s",
        json.dumps(config.to_dict(), sort_keys=True),
        seed_list,
        list(args.settings),
        bool(args.resume),
    )

    manifest_path = _write_manifest(
        output_layout=output_layout,
        args=args,
        resolved_config=config,
        seed_list=seed_list,
        settings=list(args.settings),
    )
    summary = run_figure5_experiment(
        seed_list=seed_list,
        config=config,
        output_dir=args.output_dir,
        resume=args.resume,
        settings=args.settings,
    )
    LOGGER.info("Finished Figure 5 runner summary=%s manifest=%s", output_layout.summary_path, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "preset": args.preset,
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


__all__ = ["PRESET_NUM_SEEDS", "main", "parse_args"]
