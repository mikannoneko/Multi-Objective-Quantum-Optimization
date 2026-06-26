from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from figure5_scalarization import FIGURE5_OBJECTIVE_SENSES, FIGURE5_OBJECTIVES


def is_dominated(candidate: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    """Return True when ``other`` Pareto-dominates ``candidate``.

    Figure 5 uses mixed objective directions: kappa/E are maximized and rho is
    minimized. Domination requires ``other`` to be no worse on every objective
    and strictly better on at least one objective.
    """

    candidate_values = _objective_values(candidate)
    other_values = _objective_values(other)
    no_worse = True
    strictly_better = False

    for objective in FIGURE5_OBJECTIVES:
        candidate_value = candidate_values[objective]
        other_value = other_values[objective]
        sense = FIGURE5_OBJECTIVE_SENSES[objective]
        if sense == "maximize":
            no_worse = no_worse and other_value >= candidate_value
            strictly_better = strictly_better or other_value > candidate_value
        elif sense == "minimize":
            no_worse = no_worse and other_value <= candidate_value
            strictly_better = strictly_better or other_value < candidate_value
        else:
            raise ValueError(f"Unsupported objective sense {sense!r} for {objective!r}")

    return bool(no_worse and strictly_better)


def pareto_front(points: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return all non-dominated points while preserving input order."""

    validated_points = list(points)
    for point in validated_points:
        _objective_values(point)

    front: list[Mapping[str, Any]] = []
    for candidate_index, candidate in enumerate(validated_points):
        dominated = False
        for other_index, other in enumerate(validated_points):
            if candidate_index == other_index:
                continue
            if is_dominated(candidate, other):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _objective_values(point: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for objective in FIGURE5_OBJECTIVES:
        if objective not in point:
            raise ValueError(f"point missing objective {objective!r}")
        value = float(point[objective])
        if not math.isfinite(value):
            raise ValueError(f"objective {objective!r} must be finite")
        values[objective] = value
    return values


__all__ = ["is_dominated", "pareto_front"]
