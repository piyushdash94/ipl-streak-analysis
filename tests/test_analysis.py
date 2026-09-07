"""Regression checks on the headline claims of the write-up.

Each of these locks down a number that appears in the README or notebook, so a
future change to the models cannot silently rewrite a stated conclusion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analysis


def test_v1_math_constant_is_a_denominator_error():
    """8.69% is P(valid AND 7+ wins), not P(7+ wins | valid).  The correct figure is ~24.7%."""
    table = analysis.model_comparison()
    assert table.attrs["n_all_sequences"] == 16384
    assert table.attrs["n_valid_sequences"] == 5768
    assert table.attrs["v1_denominator_check"] == pytest.approx(8.69, abs=0.01)
    at_half = table[table.p_win == 0.50].iloc[0]
    assert at_half["correct_conditional_7plus"] == pytest.approx(24.69, abs=0.05)
    assert at_half["v1_math_7plus"] == 8.69


def test_v1_simulation_reproduces_its_published_number_and_is_a_different_model():
    """v1 reported 68.2% for a strong team; the forced-reset DP gives 67.9%."""
    table = analysis.model_comparison()
    strong = table[table.p_win == 0.60].iloc[0]
    assert strong["v1_sim_7plus"] == pytest.approx(67.9, abs=0.3)
    assert strong["correct_conditional_7plus"] == pytest.approx(45.2, abs=0.3)
    # The forced-reset process is systematically more generous at every strength.
    assert (table["v1_sim_7plus"] > table["correct_conditional_7plus"]).all()


def test_streak_frequency_3plus_at_even_odds():
    """v1's app claimed '~40% of teams reach a 3+ streak' at a 50% win rate.  It is ~65%."""
    freq = analysis.streak_frequency()
    row = freq[(freq.p_win == 0.5) & (freq.streak_length == 3)].iloc[0]
    assert row["probability"] == pytest.approx(64.8, abs=0.2)
    weak = freq[(freq.p_win == 0.4) & (freq.streak_length == 3)].iloc[0]
    assert weak["probability"] == pytest.approx(42.7, abs=0.2)


def test_qualification_curve_is_monotone_and_league_size_matters():
    curves, cutoffs = analysis.format_comparison(n_seasons=6_000, seed=99)
    for n_teams, grp in curves.groupby("n_teams"):
        rates = grp.sort_values("wins")["qualify_rate"].to_numpy()
        assert np.all(np.diff(rates) >= -0.02), f"{n_teams}-team curve should rise with wins"
    eight = curves[(curves.n_teams == 8) & (curves.wins == 7)]["qualify_rate"].iloc[0]
    ten = curves[(curves.n_teams == 10) & (curves.wins == 7)]["qualify_rate"].iloc[0]
    assert eight > 0.40 and eight < 0.60, eight
    assert ten < 0.25, ten
    assert eight - ten > 0.25, "7 wins must be worth clearly less in the 10-team format"


def test_eight_wins_is_not_a_safe_ticket_in_the_ten_team_era():
    curves, _ = analysis.format_comparison(n_seasons=6_000, seed=98)
    ten = curves[(curves.n_teams == 10) & (curves.wins == 8)]["qualify_rate"].iloc[0]
    nine = curves[(curves.n_teams == 10) & (curves.wins == 9)]["qualify_rate"].iloc[0]
    assert 0.60 < ten < 0.85, ten
    assert nine > 0.95, nine


def test_confound_marginal_is_large_but_conditional_is_flat():
    out = analysis.confound_analysis(n_seasons=8_000, min_count=200)

    three = out["marginal"][out["marginal"].streak_threshold == 3].iloc[0]
    assert three["qualify_rate_with"] - three["qualify_rate_without"] > 0.40, (
        "the marginal 'streak effect' must reproduce, or the demonstration has no bite"
    )

    strat = out["stratified"]
    for wins in (7, 8):
        rates = strat[strat.wins == wins]["qualified_rate"]
        assert rates.max() - rates.min() < 0.06, (
            f"conditional on {wins} wins the streak must not matter: {rates.tolist()}"
        )

    logit = out["logit"]
    joint = logit[(logit.model == "wins + streak") & (logit.term == "max_streak")].iloc[0]
    alone = logit[(logit.model == "streak only") & (logit.term == "max_streak")].iloc[0]
    assert alone["odds_ratio"] > 2.5, "streak alone must look very predictive"
    assert abs(joint["coef"]) < 0.10, "and must collapse once wins are controlled for"
    assert joint["p_value"] > 0.01

    wins_only = logit[logit.model == "wins only"].iloc[0]["pseudo_r2"]
    both = joint["pseudo_r2"]
    assert both - wins_only < 0.01, "adding streak must not improve fit"


def test_power_curve_is_calibrated_and_powered():
    power = analysis.momentum_power_curve(gaps=(0.0, 0.15), trials=40, n_permutations=250)
    null = power[power.momentum_gap == 0.0].iloc[0]["power_at_5pct"]
    real = power[power.momentum_gap == 0.15].iloc[0]["power_at_5pct"]
    assert null < 0.20, f"false-positive rate {null}"
    assert real > 0.80, f"power against a 15-point momentum gap is only {real}"
