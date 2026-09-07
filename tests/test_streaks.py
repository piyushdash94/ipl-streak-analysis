"""Checks for src.streaks -- every DP result is verified against an
independent brute-force enumeration of all 2**n sequences."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import streaks


def brute_force_joint(n, p, cap=None):
    """Enumerate every sequence; return D[w, m] and the model-free 'conditional' pmf."""
    joint = np.zeros((n + 1, n + 1))
    for seq in itertools.product("WL", repeat=n):
        m = streaks.max_win_streak(seq)
        if cap is not None and m > cap:
            continue
        w = seq.count("W")
        joint[w, m] += p**w * (1 - p) ** (n - w)
    return joint / joint.sum()


@pytest.mark.parametrize("n", [0, 1, 5, 12])
@pytest.mark.parametrize("p", [0.3, 0.5, 0.62])
def test_unconstrained_matches_brute_force(n, p):
    dp = streaks.joint_distribution(n_matches=n, p_win=p, model="unconstrained")
    assert np.allclose(dp, brute_force_joint(n, p))


@pytest.mark.parametrize("n", [4, 10, 14])
@pytest.mark.parametrize("cap", [1, 2, 3])
@pytest.mark.parametrize("p", [0.4, 0.5, 0.6])
def test_conditional_matches_brute_force(n, cap, p):
    dp = streaks.joint_distribution(n_matches=n, p_win=p, model="conditional", cap=cap)
    assert np.allclose(dp, brute_force_joint(n, p, cap=cap))


def test_unconstrained_win_marginal_is_binomial():
    from scipy.stats import binom

    pmf = streaks.win_distribution(n_matches=14, p_win=0.6, model="unconstrained")
    assert np.allclose(pmf, binom.pmf(np.arange(15), 14, 0.6))


def test_conditional_at_half_is_uniform_over_valid_sequences():
    """At p=0.5 the conditional model is exactly enumerate-and-filter combinatorics."""
    n, cap = 14, 2
    pmf = streaks.win_distribution(n_matches=n, p_win=0.5, model="conditional", cap=cap)
    counts = np.zeros(n + 1)
    for seq in itertools.product("WL", repeat=n):
        if streaks.max_win_streak(seq) <= cap:
            counts[seq.count("W")] += 1
    assert counts.sum() == streaks.count_valid_sequences(n, cap)
    assert np.allclose(pmf, counts / counts.sum())


@pytest.mark.parametrize("cap", [1, 2, 3])
@pytest.mark.parametrize("n", [0, 1, 6, 14])
def test_count_valid_sequences_matches_enumeration(n, cap):
    expected = sum(
        1 for seq in itertools.product("WL", repeat=n) if streaks.max_win_streak(seq) <= cap
    )
    assert streaks.count_valid_sequences(n, cap) == expected


def test_count_valid_sequences_is_tribonacci_for_cap_2():
    expected = [1, 2, 4, 7, 13, 24, 44, 81, 149, 274, 504, 927, 1705, 3136, 5768]
    assert [streaks.count_valid_sequences(n, 2) for n in range(15)] == expected


def test_forced_reset_differs_from_conditional_at_p_half():
    """The v1 bug: these two were compared as if they were the same model."""
    kw = dict(n_matches=14, p_win=0.5, cap=2)
    cond = streaks.prob_at_least_wins(7, model="conditional", **kw)
    forced = streaks.prob_at_least_wins(7, model="forced_reset", **kw)
    assert not np.isclose(cond, forced, atol=0.02)


def test_forced_reset_matches_direct_simulation():
    """Independent check of the forced-reset DP against a literal re-run of the v1 loop."""
    rng = np.random.default_rng(20260907)
    n, cap, p, trials = 14, 2, 0.55, 400_000
    hits = 0
    for _ in range(trials):
        wins = run = 0
        for _ in range(n):
            if run == cap:
                run = 0
                continue
            if rng.random() < p:
                wins += 1
                run += 1
            else:
                run = 0
        hits += wins >= 7
    empirical = hits / trials
    exact = streaks.prob_at_least_wins(
        7, n_matches=n, p_win=p, model="forced_reset", cap=cap
    )
    assert abs(empirical - exact) < 0.005


def test_streak_marginal_is_monotone_in_p():
    probs = [
        streaks.prob_at_least_streak(3, n_matches=14, p_win=p, model="unconstrained")
        for p in (0.35, 0.45, 0.55, 0.65)
    ]
    assert probs == sorted(probs)


def test_max_win_streak_basics():
    assert streaks.max_win_streak([]) == 0
    assert streaks.max_win_streak("LLLL") == 0
    assert streaks.max_win_streak("LWWWL") == 3
    assert streaks.max_win_streak("WWLWWW") == 3
    assert streaks.max_win_streak(["W", "NR", "W"]) == 1


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        streaks.joint_distribution(p_win=1.5)
    with pytest.raises(ValueError):
        streaks.joint_distribution(model="conditional")
    with pytest.raises(ValueError):
        streaks.joint_distribution(model="nonsense")
