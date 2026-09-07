"""The analyses themselves, as functions returning tidy frames.

Kept separate from plotting and from the notebook so that every number in the
write-up has exactly one definition and can be regression-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import league, momentum, streaks

WIN_PROBABILITIES = (0.40, 0.50, 0.60)
V1_MATH_SEVEN_PLUS = 8.69   # the constant hard-coded in the v1 notebook and app
V1_MATH_EIGHT_PLUS = 2.49


def model_comparison(n_matches: int = 14, cap: int = 2) -> pd.DataFrame:
    """v1's two 'models' against the correct conditional model, computed exactly.

    v1 compared a combinatorial count divided by *all* 2**14 sequences against a
    simulation of a *different* process.  Both are reproduced here so the size
    of each error is visible rather than argued.
    """
    n_all = 2**n_matches
    n_valid = streaks.count_valid_sequences(n_matches, cap)
    rows = []
    for p in WIN_PROBABILITIES:
        base = dict(n_matches=n_matches, p_win=p)
        cond7 = streaks.prob_at_least_wins(7, model="conditional", cap=cap, **base)
        cond8 = streaks.prob_at_least_wins(8, model="conditional", cap=cap, **base)
        forced7 = streaks.prob_at_least_wins(7, model="forced_reset", cap=cap, **base)
        forced8 = streaks.prob_at_least_wins(8, model="forced_reset", cap=cap, **base)
        free7 = streaks.prob_at_least_wins(7, model="unconstrained", **base)
        rows.append(
            {
                "p_win": p,
                "v1_math_7plus": V1_MATH_SEVEN_PLUS,
                "v1_math_8plus": V1_MATH_EIGHT_PLUS,
                "v1_sim_7plus": 100 * forced7,
                "v1_sim_8plus": 100 * forced8,
                "correct_conditional_7plus": 100 * cond7,
                "correct_conditional_8plus": 100 * cond8,
                "unconstrained_7plus": 100 * free7,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_all_sequences"] = n_all
    out.attrs["n_valid_sequences"] = n_valid
    # v1's 8.69% is the *joint* probability P(valid AND >=7 wins) at p=0.5, i.e.
    # it divides the qualifying count by 16384 instead of by 5768.
    out.attrs["v1_denominator_check"] = 100 * (
        streaks.prob_at_least_wins(7, n_matches=n_matches, p_win=0.5, model="conditional", cap=cap)
        * n_valid
        / n_all
    )
    return out


def streak_frequency(n_matches: int = 14) -> pd.DataFrame:
    """P(longest win streak >= k) for unconstrained teams of each strength."""
    rows = []
    for p in WIN_PROBABILITIES:
        for k in range(1, 7):
            rows.append(
                {
                    "p_win": p,
                    "streak_length": k,
                    "probability": 100
                    * streaks.prob_at_least_streak(
                        k, n_matches=n_matches, p_win=p, model="unconstrained"
                    ),
                }
            )
    return pd.DataFrame(rows)


def qualification_curve(
    n_teams: int = 10,
    sigma_strength: float = 0.4,
    n_seasons: int = 30_000,
    seed: int = 20260907,
    min_count: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P(qualify | season win total), plus the distribution of the 4th-place cutoff."""
    fixtures = league.ipl_group_format() if n_teams == 10 else league.double_round_robin(n_teams)
    res = league.simulate_seasons(
        n_seasons=n_seasons, fixtures=fixtures, n_teams=n_teams,
        sigma_strength=sigma_strength, seed=seed,
    )
    df = res.to_frame()
    curve = df.groupby("wins")["qualified"].agg(n="size", qualify_rate="mean").reset_index()
    curve = curve[curve["n"] >= min_count].copy()
    curve["points"] = 2 * curve["wins"]
    curve["se"] = np.sqrt(curve.qualify_rate * (1 - curve.qualify_rate) / curve.n)
    curve["n_teams"] = n_teams

    cutoff = df[df.qualified].groupby("season")["points"].min()
    cutoff_dist = (
        cutoff.value_counts(normalize=True).sort_index().rename("share").reset_index()
    )
    cutoff_dist.columns = ["cutoff_points", "share"]
    cutoff_dist["n_teams"] = n_teams
    return curve, cutoff_dist


def format_comparison(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 8-team era against the 10-team era, four playoff spots in both."""
    curves, cutoffs = [], []
    for n_teams in (8, 10):
        curve, cutoff = qualification_curve(n_teams=n_teams, **kwargs)
        curves.append(curve)
        cutoffs.append(cutoff)
    return pd.concat(curves, ignore_index=True), pd.concat(cutoffs, ignore_index=True)


def confound_analysis(
    n_seasons: int = 40_000,
    sigma_strength: float = 0.4,
    seed: int = 20260907,
    min_count: int = 300,
) -> dict[str, pd.DataFrame]:
    """Does a longer win streak predict qualification once the win total is fixed?

    Run against a league whose match order is uniformly random, so the true
    momentum effect is exactly zero by construction.  Anything the marginal
    comparison finds here is therefore an artefact, and any honest method must
    return "no effect".
    """
    res = league.simulate_seasons(
        n_seasons=n_seasons, sigma_strength=sigma_strength, seed=seed
    )
    df = res.to_frame()

    marginal = pd.DataFrame(
        [
            {
                "streak_threshold": k,
                "share_of_teams": float((df.max_streak >= k).mean()),
                "qualify_rate_with": float(df[df.max_streak >= k].qualified.mean()),
                "qualify_rate_without": float(df[df.max_streak < k].qualified.mean()),
            }
            for k in (2, 3, 4, 5)
        ]
    )
    return {
        "marginal": marginal,
        "stratified": momentum.stratified_qualification(df, min_count=min_count),
        "logit": momentum.streak_logit(df),
        "seasons": df,
    }


def momentum_power_curve(
    n_team_seasons: int = 170,
    n_matches: int = 14,
    gaps: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20),
    trials: int = 120,
    n_permutations: int = 400,
    seed: int = 20260907,
) -> pd.DataFrame:
    """How large a momentum effect could real IPL history actually detect?

    ``n_team_seasons=170`` is roughly IPL 2008-2025.  A test with no power is
    worse than no test, because "we found nothing" then means nothing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gap in gaps:
        p_after_win, p_after_loss = 0.5 + gap / 2, 0.5 - gap / 2
        hits = 0
        for _ in range(trials):
            seqs = []
            for _ in range(n_team_seasons):
                seq, prev = [], rng.random() < 0.5
                for _ in range(n_matches):
                    prev = rng.random() < (p_after_win if prev else p_after_loss)
                    seq.append(int(prev))
                seqs.append(np.array(seq, dtype=np.int8))
            result = momentum.permutation_test(
                seqs, n_permutations=n_permutations, seed=int(rng.integers(1e9))
            )
            hits += result.p_value_greater < 0.05
        rows.append(
            {
                "momentum_gap": gap,
                "p_win_after_win": p_after_win,
                "p_win_after_loss": p_after_loss,
                "power_at_5pct": hits / trials,
            }
        )
    return pd.DataFrame(rows)


def real_data_analysis(team_seasons: pd.DataFrame, n_permutations: int = 5_000) -> dict:
    """The same questions, asked of real Cricsheet results.

    Requires the ``sequence`` column built by :func:`src.cricsheet.build_team_seasons`.
    """
    sequences = [list(s) for s in team_seasons["sequence"]]
    df = team_seasons.copy()
    df["max_streak"] = [momentum.max_streak(s) for s in sequences]
    return {
        "permutation_max_streak": momentum.permutation_test(
            sequences, momentum.mean_max_streak, n_permutations=n_permutations
        ),
        "permutation_serial_dependence": momentum.permutation_test(
            sequences, momentum.serial_dependence, n_permutations=n_permutations
        ),
        "serial_dependence": momentum.serial_dependence(sequences),
        "stratified": momentum.stratified_qualification(df, min_count=8),
        "logit": momentum.streak_logit(df),
        "qualification_by_points": df.groupby("points")["qualified"]
        .agg(n="size", qualify_rate="mean")
        .reset_index(),
    }
