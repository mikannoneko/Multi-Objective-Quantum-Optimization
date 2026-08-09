"""Shared runtime validation and reproducibility metadata for experiment runners."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence

import torch


PACKAGE_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "optuna",
    "dimod",
    "dwave-neal",
)

NEAL_SEED_MAX = (2**31) - 1
SA_SEED_ITERATION_STRIDE = 9_973


def _require_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def ensure_compute_device_available(device: str) -> None:
    """Fail before an experiment starts when its requested device is unavailable."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("Requested --device cuda, but torch.cuda.is_available() is false.")


def resolve_contiguous_seeds(default_count: int, requested_count: int | None, seed_start: int) -> list[int]:
    """Build a validated non-negative contiguous seed list."""

    count = _require_integer(
        "num_seeds",
        default_count if requested_count is None else requested_count,
    )
    start = _require_integer("seed_start", seed_start)
    if count <= 0:
        raise ValueError("num_seeds must be positive")
    if start < 0:
        raise ValueError("seed_start must be non-negative")
    final_seed = start + count - 1
    if final_seed > NEAL_SEED_MAX:
        raise ValueError(
            f"contiguous seed range ends at {final_seed}, above the neal maximum {NEAL_SEED_MAX}"
        )
    return validate_seed_list(range(start, start + count))


def validate_seed_list(
    seed_list: Sequence[int],
    *,
    iterations: int | None = None,
) -> list[int]:
    """Normalize seeds and validate the complete per-iteration SA seed schedule."""

    if len(seed_list) == 0:
        raise ValueError("seed_list must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seed_list):
        raise ValueError("seed_list must contain integers")
    normalized = [int(seed) for seed in seed_list]
    if any(seed < 0 for seed in normalized):
        raise ValueError("seeds must be non-negative")
    if any(seed > NEAL_SEED_MAX for seed in normalized):
        raise ValueError(f"seeds must not exceed the neal maximum {NEAL_SEED_MAX}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seed_list must not contain duplicates")
    if iterations is not None:
        normalized_iterations = _require_integer("iterations", iterations)
        if normalized_iterations <= 0:
            raise ValueError("iterations must be positive")
        maximum_derived_seed = max(normalized) + (
            (normalized_iterations - 1) * SA_SEED_ITERATION_STRIDE
        )
        if maximum_derived_seed > NEAL_SEED_MAX:
            raise ValueError(
                "derived SA seed exceeds the neal maximum: "
                f"max_seed + (iterations - 1) * {SA_SEED_ITERATION_STRIDE} "
                f"= {maximum_derived_seed} > {NEAL_SEED_MAX}"
            )
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(workspace_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(workspace_root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def collect_runtime_metadata(
    workspace_root: str | Path,
    reference_paper: str | Path | None = None,
) -> dict[str, Any]:
    """Collect enough environment/source identity to audit a generated result."""

    root = Path(workspace_root).resolve()
    package_versions: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None

    paper_metadata: dict[str, Any] | None = None
    if reference_paper is not None:
        paper_path = Path(reference_paper)
        if not paper_path.is_absolute():
            paper_path = root / paper_path
        paper_metadata = {
            "path": str(paper_path),
            "exists": paper_path.is_file(),
            "sha256": sha256_file(paper_path) if paper_path.is_file() else None,
        }

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": package_versions,
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": _git_metadata(root),
        "reference_paper": paper_metadata,
    }


__all__ = [
    "NEAL_SEED_MAX",
    "SA_SEED_ITERATION_STRIDE",
    "collect_runtime_metadata",
    "ensure_compute_device_available",
    "resolve_contiguous_seeds",
    "sha256_file",
    "validate_seed_list",
]
