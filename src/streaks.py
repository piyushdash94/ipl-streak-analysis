"""Exact distributions for win/loss sequences under win-streak constraints.

Everything here is computed by dynamic programming rather than Monte Carlo, so
the numbers carry no simulation error.  The DP state is

    (matches played, wins so far, current win streak, longest win streak so far)

which for a 14-match season is a few thousand states -- exhaustive enumeration
of all 2**14 sequences would also work, but the DP generalises to any season
length and any per-match win probability.

Three different processes are modelled.  They are genuinely different and the
distinction is the central methodological correction to v1 of this project:

``unconstrained``
    Every match is an independent Bernoulli(p) draw.  No streak cap.

``conditional``
    Independent Bernoulli(p) draws, *conditioned* on the event
    ``longest win streak <= cap``.  This is what "a team that never wins more
    than `cap` in a row" means as a statistical statement, and it is what
    enumerate-and-filter combinatorics computes when p = 0.5.

``forced_reset``
    After ``cap`` consecutive wins the next match is a deterministic loss.
    This is the process v1 of this project actually simulated while comparing
    it against ``conditional`` combinatorics.  It is a different process: the
    forced loss is "free" (it costs a match but consumes no coin flip), so the
    two models do not agree even at p = 0.5.  It is kept here so the size of
    the discrepancy can be quantified rather than argued about.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Model = Literal["unconstrained", "conditional", "forced_reset"]


def joint_distribution(
    n_matches: int = 14,
    p_win: float = 0.5,
    model: Model = "unconstrained",
    cap: int | None = None,
) -> np.ndarray:
    """Exact joint pmf of (total wins, longest win streak).

    Returns an array ``D`` of shape ``(n_matches + 1, n_matches + 1)`` where
    ``D[w, m]`` is P(wins = w and longest win streak = m).  It sums to 1.

    ``cap`` is required for the two constrained models and ignored otherwise.
    """
    if not 0.0 <= p_win <= 1.0:
        raise ValueError(f"p_win must be in [0, 1], got {p_win}")
    if n_matches < 0:
        raise ValueError(f"n_matches must be non-negative, got {n_matches}")
    if model in ("conditional", "forced_reset"):
        if cap is None:
            raise ValueError(f"model={model!r} requires a streak cap")
        if cap < 0:
            raise ValueError(f"cap must be non-negative, got {cap}")
    elif model != "unconstrained":
        raise ValueError(f"unknown model {model!r}")

    size = n_matches + 1
    # state[w, m, c] = probability mass on (wins=w, max streak=m, current streak=c)
    state = np.zeros((size, size, size))
    state[0, 0, 0] = 1.0

    for _ in range(n_matches):
        nxt = np.zeros_like(state)
        nz = np.argwhere(state > 0.0)
        for w, m, c in nz:
            mass = state[w, m, c]

            if model == "forced_reset" and c == cap:
                # Deterministic loss: no coin flip is spent on this match.
                nxt[w, m, 0] += mass
                continue

            # Loss branch.
            nxt[w, m, 0] += mass * (1.0 - p_win)

            # Win branch.
            c2 = c + 1
            if model in ("conditional", "forced_reset") and c2 > cap:
                # `conditional`: this path violates the constraint, so its mass
                # is discarded and the survivors are renormalised below.
                # (`forced_reset` never reaches here -- it is caught above.)
                continue
            nxt[w + 1, max(m, c2), c2] += mass * p_win

        state = nxt

    joint = state.sum(axis=2)
    total = joint.sum()
    if total <= 0.0:
        raise ValueError(
            "no sequences satisfy the constraint "
            f"(n_matches={n_matches}, cap={cap}, p_win={p_win})"
        )
    return joint / total  # renormalisation is a no-op unless model == 'conditional'


def win_distribution(**kwargs) -> np.ndarray:
    """Marginal pmf of total wins.  Same arguments as :func:`joint_distribution`."""
    return joint_distribution(**kwargs).sum(axis=1)


def max_streak_distribution(**kwargs) -> np.ndarray:
    """Marginal pmf of the longest win streak.  Same args as :func:`joint_distribution`."""
    return joint_distribution(**kwargs).sum(axis=0)


def prob_at_least_wins(threshold: int, **kwargs) -> float:
    """P(total wins >= threshold).  Same arguments as :func:`joint_distribution`."""
    return float(win_distribution(**kwargs)[threshold:].sum())


def prob_at_least_streak(length: int, **kwargs) -> float:
    """P(longest win streak >= length).  Same args as :func:`joint_distribution`."""
    return float(max_streak_distribution(**kwargs)[length:].sum())


def count_valid_sequences(n_matches: int = 14, cap: int = 2) -> int:
    """Number of W/L sequences of length ``n_matches`` with no run of wins > ``cap``.

    Satisfies the ``cap``-step recurrence a(n) = a(n-1) + ... + a(n-cap-1);
    for cap=2 this is the tribonacci sequence.  Computed directly here so the
    combinatorial count can be checked against the DP independently.
    """
    if cap < 0 or n_matches < 0:
        raise ValueError("n_matches and cap must be non-negative")
    # counts[c] = number of prefixes ending in a current win streak of length c
    counts = [0] * (cap + 1)
    counts[0] = 1
    for _ in range(n_matches):
        nxt = [0] * (cap + 1)
        nxt[0] = sum(counts)          # append a loss
        for c in range(cap):
            nxt[c + 1] = counts[c]     # append a win
        counts = nxt
    return sum(counts)


def max_win_streak(sequence) -> int:
    """Longest run of ``'W'`` in an iterable of match results."""
    best = run = 0
    for outcome in sequence:
        if outcome == "W":
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best
