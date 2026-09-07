"""Testing whether win streaks carry information, or just reflect win totals.

The claim v1 of this project ended on was: *teams that build 3+ match win
streaks qualify; teams that do not, mostly fail.*  That is descriptively true
and analytically empty, because a team's longest win streak is mechanically
driven by how many matches it won.  A 10-win team almost cannot avoid a 3-match
streak; a 4-win team almost cannot produce one.  So "streak teams qualify more"
may be nothing more than "teams that win more matches qualify more" wearing a
disguise.

Two tools separate the two stories.

1. **Condition on the win total.**  Compare teams that won the *same* number of
   matches and differ only in how those wins were arranged.  If the streak
   still predicts qualification there, it carries real information.
   :func:`stratified_qualification` and :func:`streak_logit` do this.

2. **Permute within season.**  Reshuffling a team's own 14 results preserves its
   win total exactly and destroys the ordering.  Comparing the observed streak
   statistic against that null answers "are IPL streaks longer than chance
   would produce for teams of this quality?" without assuming any win
   probability.  :func:`permutation_test` does this.

This is the standard hot-hand design, and it is the part v1 was missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------


def as_indicator(sequence: Sequence[str] | Sequence[int]) -> np.ndarray:
    """Coerce a result sequence to a 0/1 array (1 = win).  Non-'W' counts as 0.

    Accepts ``"LWWWL"``, ``["L", "W", "NR"]`` or ``[0, 1, 1]``.  Anything that
    is not ``'W'`` -- a loss, a no-result, an abandoned match -- is a 0, which
    is the right treatment for a *win* streak.
    """
    if isinstance(sequence, str):
        sequence = list(sequence)
    arr = np.asarray(sequence)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.kind in "iub":
        return (arr != 0).astype(np.int8)
    return (arr == "W").astype(np.int8)


def max_streak(sequence) -> int:
    x = as_indicator(sequence)
    best = run = 0
    for v in x:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def n_runs(sequence) -> int:
    """Number of maximal same-result blocks.  Few runs = clumped = 'streaky'."""
    x = as_indicator(sequence)
    if x.size == 0:
        return 0
    return int(1 + (x[1:] != x[:-1]).sum())


def runs_test_z(sequence) -> float:
    """Wald-Wolfowitz z for too few / too many runs, conditional on the win total.

    Negative z = fewer runs than chance = clumped results = apparent momentum.
    Returns ``nan`` when the sequence is all wins or all losses.
    """
    x = as_indicator(sequence)
    n = x.size
    n1, n0 = int(x.sum()), int(n - x.sum())
    if n1 == 0 or n0 == 0 or n < 2:
        return float("nan")
    mean = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n**2 * (n - 1))
    if var <= 0:
        return float("nan")
    return (n_runs(x) - mean) / np.sqrt(var)


def serial_dependence(sequences: Sequence[Sequence]) -> float:
    """Pooled P(win | previous win) - P(win | previous loss).

    Positive = wins follow wins more often than they follow losses.  Note the
    null value is slightly *negative*, not zero, because sampling a fixed set of
    results without replacement induces mild negative dependence -- which is
    exactly why this should be judged against a permutation null rather than 0.
    """
    after_win = after_win_n = after_loss = after_loss_n = 0
    for seq in sequences:
        x = as_indicator(seq).astype(np.int64)
        if x.size < 2:
            continue
        prev, cur = x[:-1], x[1:]
        after_win += int(cur[prev == 1].sum())
        after_win_n += int((prev == 1).sum())
        after_loss += int(cur[prev == 0].sum())
        after_loss_n += int((prev == 0).sum())
    if after_win_n == 0 or after_loss_n == 0:
        return float("nan")
    return after_win / after_win_n - after_loss / after_loss_n


def mean_max_streak(sequences: Sequence[Sequence]) -> float:
    return float(np.mean([max_streak(s) for s in sequences]))


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null_mean: float
    null_sd: float
    p_value_greater: float
    p_value_two_sided: float
    n_permutations: int

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"observed={self.observed:.4f}  null={self.null_mean:.4f}"
            f" (sd {self.null_sd:.4f})  p(one-sided)={self.p_value_greater:.4f}"
            f"  p(two-sided)={self.p_value_two_sided:.4f}"
        )


def permutation_test(
    sequences: Sequence[Sequence],
    statistic: Callable[[Sequence[Sequence]], float] = mean_max_streak,
    n_permutations: int = 5_000,
    seed: int | None = 20260907,
) -> PermutationResult:
    """Shuffle each sequence within itself; compare the statistic to that null.

    Each team-season keeps its exact win total, so the test asks only about
    *ordering*.  It makes no assumption about per-match win probability and
    needs no model of team strength.
    """
    rng = np.random.default_rng(seed)
    arrays = [as_indicator(s) for s in sequences]
    observed = statistic(arrays)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = statistic([rng.permutation(a) for a in arrays])
    # +1 in numerator and denominator: the observed value is itself a valid
    # draw under the null, which keeps the p-value from ever being exactly 0.
    p_greater = (1 + int((null >= observed).sum())) / (n_permutations + 1)
    p_two = (1 + int((np.abs(null - null.mean()) >= abs(observed - null.mean())).sum())) / (
        n_permutations + 1
    )
    return PermutationResult(
        observed=float(observed),
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)),
        p_value_greater=float(p_greater),
        p_value_two_sided=float(p_two),
        n_permutations=n_permutations,
    )


# ---------------------------------------------------------------------------
# Conditioning on the win total
# ---------------------------------------------------------------------------


def stratified_qualification(
    df: pd.DataFrame,
    wins_col: str = "wins",
    streak_col: str = "max_streak",
    qualified_col: str = "qualified",
    min_count: int = 30,
) -> pd.DataFrame:
    """Qualification rate by (win total, longest streak), with counts.

    Rows thinner than ``min_count`` are dropped -- with a binary outcome, a
    cell of five teams tells you nothing and invites over-reading.
    """
    grouped = (
        df.groupby([wins_col, streak_col])[qualified_col]
        .agg(n="size", qualified_rate="mean")
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_count].copy()
    # Wilson-style standard error is enough for eyeballing overlap.
    rate = grouped["qualified_rate"]
    grouped["se"] = np.sqrt(rate * (1 - rate) / grouped["n"])
    return grouped.sort_values([wins_col, streak_col]).reset_index(drop=True)


def streak_logit(
    df: pd.DataFrame,
    wins_col: str = "wins",
    streak_col: str = "max_streak",
    qualified_col: str = "qualified",
) -> pd.DataFrame:
    """Logistic regressions of qualification on streak alone, then with wins added.

    The comparison is the point: a large streak coefficient that collapses once
    the win total enters the model is a confound, not an effect.
    """
    import statsmodels.api as sm

    y = df[qualified_col].astype(float).to_numpy()
    out = []
    specs = {
        "streak only": [streak_col],
        "wins only": [wins_col],
        "wins + streak": [wins_col, streak_col],
    }
    for name, cols in specs.items():
        X = sm.add_constant(df[cols].astype(float).to_numpy(), has_constant="add")
        model = sm.Logit(y, X).fit(disp=0)
        for j, col in enumerate(cols, start=1):
            out.append(
                {
                    "model": name,
                    "term": col,
                    "coef": model.params[j],
                    "std_err": model.bse[j],
                    "z": model.tvalues[j],
                    "p_value": model.pvalues[j],
                    "odds_ratio": float(np.exp(model.params[j])),
                    "pseudo_r2": model.prsquared,
                }
            )
    return pd.DataFrame(out)
