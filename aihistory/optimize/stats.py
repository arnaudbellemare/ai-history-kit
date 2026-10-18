"""Paired comparison of two arms over the same prompts."""

from __future__ import annotations

import math


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)


def majority_correct(scores: list[float]) -> bool:
    return bool(scores) and sum(scores) / len(scores) > 0.5


def paired(per_prompt_a: dict[str, list[float]], per_prompt_b: dict[str, list[float]]) -> dict:
    """Prompts each arm gets right (by majority of runs) and the other gets wrong."""
    shared = [p for p in per_prompt_a if p in per_prompt_b]
    only_a = sum(1 for p in shared if majority_correct(per_prompt_a[p]) and not majority_correct(per_prompt_b[p]))
    only_b = sum(1 for p in shared if majority_correct(per_prompt_b[p]) and not majority_correct(per_prompt_a[p]))
    return {"prompts": len(shared), "only_a": only_a, "only_b": only_b, "p": round(mcnemar_exact(only_a, only_b), 3)}
