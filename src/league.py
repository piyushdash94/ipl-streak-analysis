"""Whole-league IPL season simulator.

v1 of this project simulated a *single* team in isolation with a fixed win
probability.  That has a structural problem the v1 notebook itself flagged but
never fixed: in a real league every match has a winner and a loser, so wins are
conserved (total points across the table is fixed) and qualification is
*relative* -- you do not qualify by hitting 14 points, you qualify by finishing
in the top four of a table that has to add up.

This module simulates the full table.  Team strengths are Bradley-Terry:

    P(i beats j) = sigma(theta_i - theta_j),  theta ~ Normal(0, sigma_strength)

``sigma_strength = 0`` reproduces the coin-flip league exactly; larger values
spread the table out.  Because it is not currently possible to fit
``sigma_strength`` to real IPL results from this environment (see
``src/cricsheet.py``), every headline result should be reported across a range
of values rather than at one calibrated point.

Crucially for the streak question: each team's 14 results are placed in a
*uniformly random order*.  There is therefore **exactly zero momentum** in the
data-generating process.  Any apparent "streaks predict qualification" signal
that shows up in this simulation is a mechanical artefact of the fact that
teams which win more matches also, unavoidably, produce longer runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def double_round_robin(n_teams: int) -> np.ndarray:
    """Every pair meets twice.  The IPL format for 8 teams (2008-10, 2014-21)."""
    fixtures = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
    return np.array(fixtures, dtype=np.int16)


def ipl_group_format() -> np.ndarray:
    """The 10-team format used since 2022: 70 league matches, 14 per team.

    Teams 0-4 are group A, teams 5-9 group B.  Each team plays everyone in its
    own group twice (8 matches), one designated cross-group rival twice (2),
    and the other four cross-group teams once (4).
    """
    group_a, group_b = list(range(5)), list(range(5, 10))
    fixtures: list[tuple[int, int]] = []
    for group in (group_a, group_b):
        for i in group:
            for j in group:
                if i != j:
                    fixtures.append((i, j))
    for k, i in enumerate(group_a):
        rival = group_b[k]
        fixtures.append((i, rival))
        fixtures.append((rival, i))  # the designated rivalry, home and away
        for j in group_b:
            if j != rival:
                fixtures.append((i, j) if (k + j) % 2 == 0 else (j, i))
    return np.array(fixtures, dtype=np.int16)


def matches_per_team(fixtures: np.ndarray, n_teams: int) -> np.ndarray:
    return np.bincount(fixtures.ravel(), minlength=n_teams)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonResults:
    """One row per team per simulated season (flattened: ``n_seasons * n_teams``)."""

    season: np.ndarray      # season index
    team: np.ndarray        # team index
    wins: np.ndarray
    no_results: np.ndarray
    points: np.ndarray
    nrr: np.ndarray         # proxy: mean signed match margin (see module docstring)
    max_streak: np.ndarray  # longest win streak, results in uniformly random order
    position: np.ndarray    # 1 = top of the table
    qualified: np.ndarray   # bool: finished in the top ``playoff_spots``
    strength: np.ndarray    # the Bradley-Terry theta the team was given

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame({f: getattr(self, f) for f in self.__dataclass_fields__})


def _longest_run(indicator: np.ndarray) -> np.ndarray:
    """Longest run of 1s along axis 1, vectorised over rows."""
    running = np.zeros(indicator.shape[0], dtype=np.int16)
    best = np.zeros(indicator.shape[0], dtype=np.int16)
    for col in range(indicator.shape[1]):
        running = (running + 1) * indicator[:, col]
        np.maximum(best, running, out=best)
    return best


def _random_order_max_streak(wins: np.ndarray, n_matches: int, rng) -> np.ndarray:
    """Longest win streak when each team's results are shuffled uniformly.

    Marginally, a uniformly random fixture order gives each team a uniformly
    random arrangement of its own results, so this is exact -- and it encodes
    the *no-momentum null* the streak analysis is tested against.
    """
    ranks = rng.random((wins.size, n_matches)).argsort(axis=1)
    indicator = (ranks < wins[:, None]).astype(np.int16)
    return _longest_run(indicator)


def simulate_seasons(
    n_seasons: int = 20_000,
    fixtures: np.ndarray | None = None,
    n_teams: int = 10,
    sigma_strength: float = 0.4,
    playoff_spots: int = 4,
    p_no_result: float = 0.0,
    seed: int | None = 20260907,
) -> SeasonResults:
    """Simulate ``n_seasons`` complete league campaigns.

    ``sigma_strength`` is the SD of the Bradley-Terry log-strengths; 0 gives a
    league of identical coin-flipping teams.  ``p_no_result`` is the chance a
    match is abandoned (1 point each).
    """
    rng = np.random.default_rng(seed)
    if fixtures is None:
        fixtures = ipl_group_format() if n_teams == 10 else double_round_robin(n_teams)
    per_team = matches_per_team(fixtures, n_teams)
    if len(set(per_team.tolist())) != 1:
        raise ValueError(f"schedule is unbalanced: matches per team = {per_team.tolist()}")
    n_matches = int(per_team[0])
    home, away = fixtures[:, 0].astype(np.int64), fixtures[:, 1].astype(np.int64)
    n_fixtures = fixtures.shape[0]

    theta = rng.normal(0.0, sigma_strength, size=(n_seasons, n_teams))
    p_home = 1.0 / (1.0 + np.exp(-(theta[:, home] - theta[:, away])))

    home_wins = rng.random((n_seasons, n_fixtures)) < p_home
    abandoned = (
        rng.random((n_seasons, n_fixtures)) < p_no_result
        if p_no_result > 0
        else np.zeros((n_seasons, n_fixtures), dtype=bool)
    )
    # Margin proxy: half-normal magnitude, signed towards the winner.
    margin = np.abs(rng.normal(0.0, 1.0, size=(n_seasons, n_fixtures)))
    margin[abandoned] = 0.0

    wins = np.zeros((n_seasons, n_teams), dtype=np.int16)
    no_results = np.zeros((n_seasons, n_teams), dtype=np.int16)
    margin_sum = np.zeros((n_seasons, n_teams))
    played = np.zeros((n_seasons, n_teams), dtype=np.int16)

    live = ~abandoned
    for side, won_mask in ((home, home_wins & live), (away, (~home_wins) & live)):
        np.add.at(wins.T, side, won_mask.T)
    for side in (home, away):
        np.add.at(no_results.T, side, abandoned.T)
        np.add.at(played.T, side, np.ones_like(abandoned).T)
    np.add.at(margin_sum.T, home, np.where(home_wins & live, margin, -margin).T)
    np.add.at(margin_sum.T, away, np.where((~home_wins) & live, margin, -margin).T)

    points = 2 * wins.astype(np.int32) + no_results.astype(np.int32)
    nrr = margin_sum / np.maximum(played, 1)

    # Rank on points, then NRR.  argsort twice turns a sort key into a position.
    key = -(points.astype(np.float64) * 1e6 + nrr)
    order = np.argsort(key, axis=1, kind="stable")
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(1, n_teams + 1)[None, :].repeat(n_seasons, 0), axis=1)
    qualified = position <= playoff_spots

    flat_wins = wins.ravel()
    max_streak = _random_order_max_streak(flat_wins.astype(np.int64), n_matches, rng)

    season_idx = np.repeat(np.arange(n_seasons), n_teams)
    team_idx = np.tile(np.arange(n_teams), n_seasons)
    return SeasonResults(
        season=season_idx,
        team=team_idx,
        wins=flat_wins,
        no_results=no_results.ravel(),
        points=points.ravel(),
        nrr=nrr.ravel(),
        max_streak=max_streak,
        position=position.ravel(),
        qualified=qualified.ravel(),
        strength=theta.ravel(),
    )
