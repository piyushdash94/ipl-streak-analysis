"""Invariant checks for the whole-league simulator.

The point of these is that a league simulation is only worth anything if the
table adds up: wins are conserved, exactly four teams qualify, positions are a
permutation.  v1's single-team model could not satisfy any of these because it
never modelled an opponent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import league


def test_group_format_shape():
    fx = league.ipl_group_format()
    assert fx.shape == (70, 2)
    assert (league.matches_per_team(fx, 10) == 14).all()
    assert not (fx[:, 0] == fx[:, 1]).any()


def test_group_format_pairings():
    """8 intra-group matches, 6 cross-group, exactly one cross-group rival twice."""
    fx = league.ipl_group_format()
    for team in range(10):
        own = set(range(5)) if team < 5 else set(range(5, 10))
        opponents = [b if a == team else a for a, b in fx if team in (a, b)]
        intra = [o for o in opponents if o in own]
        cross = [o for o in opponents if o not in own]
        assert len(intra) == 8 and len(set(intra)) == 4
        assert len(cross) == 6 and len(set(cross)) == 5
        twice = [o for o in set(cross) if cross.count(o) == 2]
        assert len(twice) == 1


def test_double_round_robin_shape():
    fx = league.double_round_robin(8)
    assert fx.shape == (56, 2)
    assert (league.matches_per_team(fx, 8) == 14).all()


def test_unbalanced_schedule_rejected():
    with pytest.raises(ValueError, match="unbalanced"):
        league.simulate_seasons(n_seasons=2, fixtures=np.array([[0, 1], [0, 2]]), n_teams=3)


@pytest.mark.parametrize("sigma", [0.0, 0.4, 0.9])
def test_results_are_conserved(sigma):
    n_seasons = 300
    res = league.simulate_seasons(n_seasons=n_seasons, sigma_strength=sigma, seed=1)
    df = res.to_frame()
    per_season = df.groupby("season")[["wins", "points"]].sum()
    assert (per_season["wins"] == 70).all(), "one winner per match"
    assert (per_season["points"] == 140).all(), "2 points per match, always"


def test_no_results_still_conserve_points():
    res = league.simulate_seasons(n_seasons=300, p_no_result=0.05, seed=2)
    df = res.to_frame()
    per_season = df.groupby("season").agg(w=("wins", "sum"), nr=("no_results", "sum"), p=("points", "sum"))
    assert (2 * per_season["w"] + per_season["nr"] == 140).all()
    assert (per_season["p"] == 140).all()
    assert per_season["nr"].sum() > 0, "the no-result path should actually fire"


def test_positions_and_qualification():
    res = league.simulate_seasons(n_seasons=400, seed=3)
    df = res.to_frame()
    for _, grp in df.groupby("season"):
        assert sorted(grp["position"]) == list(range(1, 11))
        assert grp["qualified"].sum() == 4
        top = grp.sort_values("position")
        # position must be consistent with the (points, nrr) sort key
        keys = list(zip(-top["points"], -top["nrr"]))
        assert keys == sorted(keys)


def test_equal_strength_league_is_binomial():
    res = league.simulate_seasons(n_seasons=4000, sigma_strength=0.0, seed=4)
    counts = np.bincount(res.wins, minlength=15)[:15]
    expected = binom.pmf(np.arange(15), 14, 0.5) * res.wins.size
    # chi-square-ish tolerance on the bulk of the distribution
    mask = expected > 30
    assert np.all(np.abs(counts[mask] - expected[mask]) < 4.5 * np.sqrt(expected[mask]))


def test_stronger_teams_qualify_more():
    res = league.simulate_seasons(n_seasons=3000, sigma_strength=0.6, seed=5)
    df = res.to_frame()
    lo = df[df.strength < df.strength.quantile(0.25)]["qualified"].mean()
    hi = df[df.strength > df.strength.quantile(0.75)]["qualified"].mean()
    assert hi > lo + 0.20


def test_max_streak_bounds():
    res = league.simulate_seasons(n_seasons=1000, seed=6)
    assert (res.max_streak <= res.wins).all()
    assert (res.max_streak >= (res.wins > 0)).all()
    assert (res.max_streak[res.wins == 14] == 14).all()


def test_longest_run_reference():
    rng = np.random.default_rng(7)
    x = (rng.random((500, 14)) < 0.5).astype(np.int16)

    def slow(row):
        best = run = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    assert (league._longest_run(x) == np.array([slow(r) for r in x])).all()


def test_random_order_streak_matches_exact_hypergeometric_dp():
    """Shuffled-order max streak must match the exact combinatorial distribution."""
    from src import streaks

    rng = np.random.default_rng(8)
    n, w = 14, 7
    sample = league._random_order_max_streak(np.full(200_000, w), n, rng)
    empirical = np.bincount(sample, minlength=n + 1)[: n + 1] / sample.size
    # Exact: uniform over the C(14,7) arrangements -> conditional on wins under p=0.5
    joint = streaks.joint_distribution(n_matches=n, p_win=0.5, model="unconstrained")
    exact = joint[w] / joint[w].sum()
    assert np.abs(empirical - exact).max() < 0.005


def test_seed_is_reproducible():
    a = league.simulate_seasons(n_seasons=50, seed=42).to_frame()
    b = league.simulate_seasons(n_seasons=50, seed=42).to_frame()
    assert a.equals(b)
