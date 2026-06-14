from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import torch

from figure4_pipeline import Figure4Config, OBJECTIVES, SUPPORTED_SETTINGS, ensure_runtime_dependencies, run_figure4_experiment


SUPPORTED_INTERPRETER_FRAGMENT = "\\env_torch\\"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Figure 4 PyTorch FM reproduction pipeline.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary outputs.")
    # Paper-scale defaults used by Figure4Config:
    #   --num-seeds 20 --iterations 600 --num-samples 100 --num-levels 50
    #   --optuna-trials 20 --sa-runs 1000 --sa-sweeps 3000
    # The current local Figure 4 reproduction was run at reduced scale:
    #   w/o CGFM equivalent now requires:
    #     python.exe figure4_runner.py --output-dir figure4_quick_l50 --device cuda --resume
    #       --settings wo_cgfm --num-seeds 3 --iterations 100 --num-levels 50
    #       --num-samples 100 --optuna-trials 3 --sa-runs 100 --sa-sweeps 500
    #   w/ CGFM:
    #     python.exe figure4_runner.py --output-dir figure4_cgfm_quick_l50 --device cuda --resume
    #       --settings w_cgfm --num-seeds 3 --iterations 100 --num-levels 50
    #       --num-samples 100 --optuna-trials 3 --sa-runs 100 --sa-sweeps 500
    # Compared with the paper-scale defaults, only num_samples and num_levels are unchanged.
    # The quick run reduces seeds, active-learning iterations, FM tuning trials, and SA effort.
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of initial datasets / trajectories.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the contiguous seed list.")
    parser.add_argument("--iterations", type=int, default=600, help="Active learning iterations per trajectory.")
    parser.add_argument("--num-samples", type=int, default=100, help="Initial dataset size per seed.")
    parser.add_argument("--num-levels", type=int, default=50, help="One-hot levels per variable block.")
    parser.add_argument("--optuna-trials", type=int, default=20, help="Number of Optuna trials for FM tuning.")
    parser.add_argument("--sa-runs", type=int, default=1000, help="Simulated annealing restart count.")
    parser.add_argument("--sa-sweeps", type=int, default=3000, help="Simulated annealing sweep count.")
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


def _ensure_env_torch() -> None:
    executable = sys.executable.lower().replace("/", "\\")
    if SUPPORTED_INTERPRETER_FRAGMENT not in executable:
        raise EnvironmentError(
            "Use the env_torch interpreter: "
            r"C:\Users\13303\anaconda3\envs\env_torch\python.exe"
        )


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    config: Figure4Config,
    seed_list: list[int],
    objective_names: list[str],
    settings: list[str],
) -> Path:
    manifest = {
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": asdict(config),
        "seed_list": seed_list,
        "objectives": objective_names,
        "settings": settings,
        "resume": bool(args.resume),
        "summary_path": str(output_dir / "figure4_summary.json"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    _ensure_env_torch()
    ensure_runtime_dependencies()
    args = parse_args()
    selected_objectives = OBJECTIVES
    if args.objectives:
        requested = set(args.objectives)
        selected_objectives = tuple(objective for objective in OBJECTIVES if objective.name in requested)
        if not selected_objectives:
            raise ValueError("No valid objectives selected.")

    config = Figure4Config(
        num_samples=args.num_samples,
        iterations=args.iterations,
        num_levels=args.num_levels,
        optuna_trials=args.optuna_trials,
        sa_runs=args.sa_runs,
        sa_sweeps=args.sa_sweeps,
        device=args.device,
    )
    seed_list = list(range(args.seed_start, args.seed_start + args.num_seeds))
    summary = run_figure4_experiment(
        seed_list=seed_list,
        config=config,
        output_dir=args.output_dir,
        objectives=selected_objectives,
        resume=args.resume,
        settings=args.settings,
    )
    manifest_path = _write_manifest(
        output_dir=args.output_dir,
        args=args,
        config=config,
        seed_list=seed_list,
        objective_names=[obj.name for obj in selected_objectives],
        settings=list(args.settings),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "objectives": [obj.name for obj in selected_objectives],
                "settings": list(args.settings),
                "training_backend": summary["training_backend"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
