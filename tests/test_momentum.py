"""Checks for the momentum tests, including calibration and power.

A hypothesis test is only trustworthy if you have shown (a) it does not fire on
data with no effect, and (b) it does fire when an effect is planted.  Both are
checked here so the test can be believed when it is pointed at real data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import momentum


def iid_sequences(n_seq, n_matches, p, rng):
    return [(rng.random(n_matches) < p).astype(np.int8) for _ in range(n_seq)]


def markov_sequences(n_seq, n_matches, p_after_win, p_after_loss, rng):
    """Sequences with genuine momentum: winning raises the next-match win chance."""
    out = []
    for _ in range(n_seq):
        seq, prev = [], rng.random() < 0.5
        for _ in range(n_matches):
            p = p_after_win if prev else p_after_loss
            prev = rng.random() < p
            seq.append(int(prev))
        out.append(np.array(seq, dtype=np.int8))
    return out


def test_max_streak_and_runs():
    assert momentum.max_streak("LWWWL") == 3
    assert momentum.max_streak([1, 1, 0, 1]) == 2
    assert momentum.max_streak("") == 0
    assert momentum.n_runs("WWLL") == 2
    assert momentum.n_runs("WLWLW") == 5
    assert momentum.n_runs("") == 0


def test_runs_test_z_hand_computed():
    # n=4, n1=n0=2, runs=2 -> mean 3, var 2/3, z = -1/sqrt(2/3)
    assert momentum.runs_test_z("WWLL") == pytest.approx(-1.0 / np.sqrt(2 / 3), rel=1e-9)
    assert np.isnan(momentum.runs_test_z("WWWW"))
    assert np.isnan(momentum.runs_test_z("L"))


def test_as_indicator_treats_no_result_as_not_a_win():
    assert list(momentum.as_indicator(["W", "NR", "L", "W"])) == [1, 0, 0, 1]


def test_permutation_null_is_calibrated_on_iid_data():
    """No momentum in the data -> p-values should be roughly uniform, not small."""
    rng = np.random.default_rng(11)
    p_values = []
    for _ in range(40):
        seqs = iid_sequences(60, 14, 0.5, rng)
        p_values.append(
            momentum.permutation_test(seqs, n_permutations=400, seed=int(rng.integers(1e6))).p_value_two_sided
        )
    reject_rate = np.mean(np.array(p_values) < 0.05)
    assert reject_rate <= 0.15, f"false-positive rate too high: {reject_rate}"


def test_permutation_test_detects_planted_momentum():
    rng = np.random.default_rng(12)
    seqs = markov_sequences(180, 14, p_after_win=0.68, p_after_loss=0.32, rng=rng)
    res = momentum.permutation_test(seqs, n_permutations=2000, seed=13)
    assert res.observed > res.null_mean
    assert res.p_value_greater < 0.01, res


def test_permutation_test_preserves_win_totals():
    """The null must not change any team's record -- only the ordering."""
    rng = np.random.default_rng(14)
    seqs = iid_sequences(30, 14, 0.5, rng)
    totals = sorted(int(s.sum()) for s in seqs)
    permuted = [np.random.default_rng(15).permutation(s) for s in seqs]
    assert sorted(int(s.sum()) for s in permuted) == totals


def test_serial_dependence_signs():
    rng = np.random.default_rng(16)
    momentum_seqs = markov_sequences(400, 14, 0.7, 0.3, rng)
    iid_seqs = iid_sequences(400, 14, 0.5, rng)
    assert momentum.serial_dependence(momentum_seqs) > 0.25
    assert abs(momentum.serial_dependence(iid_seqs)) < 0.08


def test_stratified_qualification_drops_thin_cells():
    df = pd.DataFrame(
        {
            "wins": [7] * 100 + [8] * 5,
            "max_streak": [2] * 50 + [3] * 50 + [4] * 5,
            "qualified": [False] * 50 + [True] * 50 + [True] * 5,
        }
    )
    out = momentum.stratified_qualification(df, min_count=30)
    assert set(out["wins"]) == {7}
    assert len(out) == 2
    assert out["n"].tolist() == [50, 50]


def test_streak_logit_recovers_a_planted_effect():
    """Sanity check that the regression can see a real streak effect when one exists."""
    rng = np.random.default_rng(17)
    n = 8000
    wins = rng.integers(3, 12, n)
    streak = np.clip((wins / 2 + rng.normal(0, 1, n)).round(), 1, 10).astype(int)
    logit = -6 + 0.7 * wins + 0.5 * streak
    qualified = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({"wins": wins, "max_streak": streak, "qualified": qualified})
    out = momentum.streak_logit(df)
    joint = out[(out.model == "wins + streak") & (out.term == "max_streak")].iloc[0]
    assert joint["coef"] == pytest.approx(0.5, abs=0.15)
    assert joint["p_value"] < 1e-4
