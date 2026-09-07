"""Interactive companion to the IPL streak analysis.

Every number on this page is computed live by ``src/`` -- nothing is
hard-coded.  That is deliberate: the previous version of this app carried
pasted constants that had drifted away from (and in places never matched) the
notebook they came from.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import analysis, momentum, streaks  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

st.set_page_config(page_title="IPL Streaks & Playoff Qualification", layout="wide")
st.title("🏏 IPL: what a win streak is actually worth")
st.caption(
    "A rebuild of the original streak analysis. Two of its three headline numbers "
    "were arithmetic or modelling errors; the third was a confound. Details below."
)


@st.cache_data
def _model_table():
    return analysis.model_comparison()


@st.cache_data
def _curves(n_seasons: int):
    return analysis.format_comparison(n_seasons=n_seasons)


@st.cache_data
def _confound(n_seasons: int):
    out = analysis.confound_analysis(n_seasons=n_seasons)
    return out["marginal"], out["stratified"], out["logit"]


n_seasons = st.sidebar.slider("Simulated seasons", 5_000, 60_000, 20_000, step=5_000)
st.sidebar.caption("More seasons = tighter estimates, slower first run.")

# ---------------------------------------------------------------- section 1
st.header("1. The points table, and what it is actually worth")
st.markdown(
    "The old version assumed *7 wins gives you a chance, 8 wins is basically safe*. "
    "That assumption came from the 8-team era. Since 2022 there are ten teams "
    "competing for the same four playoff places, and the arithmetic changed."
)
curves, cutoffs = _curves(n_seasons)
fig = go.Figure()
for colour, (n_teams, grp) in zip(SERIES, curves.groupby("n_teams")):
    grp = grp[grp.wins.between(4, 12)]
    fig.add_trace(
        go.Scatter(
            x=grp.wins, y=grp.qualify_rate * 100, mode="lines+markers",
            name=f"{n_teams} teams, 4 spots", line=dict(width=3, color=colour),
            marker=dict(size=9),
        )
    )
fig.update_layout(
    xaxis_title="League-stage wins (of 14)", yaxis_title="Chance of finishing top four (%)",
    height=480, legend_title=None,
)
st.plotly_chart(fig, use_container_width=True)

ten = curves[curves.n_teams == 10].set_index("wins")["qualify_rate"]
eight = curves[curves.n_teams == 8].set_index("wins")["qualify_rate"]
cols = st.columns(3)
cols[0].metric("7 wins, 8-team era", f"{eight.get(7, float('nan'))*100:.0f}%")
cols[1].metric("7 wins, 10-team era", f"{ten.get(7, float('nan'))*100:.0f}%",
               delta=f"{(ten.get(7, 0)-eight.get(7, 0))*100:.0f} pts")
cols[2].metric("8 wins, 10-team era", f"{ten.get(8, float('nan'))*100:.0f}%")
st.markdown(
    f"Most common 4th-place cutoff in the 10-team format: "
    f"**{int(cutoffs[cutoffs.n_teams == 10].sort_values('share').iloc[-1]['cutoff_points'])} points**. "
    "Sixteen points is where the traffic jam is, which is exactly why NRR keeps deciding it."
)

# ---------------------------------------------------------------- section 2
st.header("2. Do streaks predict qualification, or just reflect wins?")
st.markdown(
    "Below is a simulated league in which match order is **uniformly random** — "
    "there is provably zero momentum in it. If the 'streak effect' shows up here "
    "anyway, it is not evidence of momentum."
)
marginal, stratified, logit = _confound(n_seasons)
left, right = st.columns(2)

with left:
    st.subheader("Pooled across all teams")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{int(k)}+ in a row" for k in marginal.streak_threshold],
                         y=marginal.qualify_rate_with * 100, name="with the streak",
                         marker_color=SERIES[0]))
    fig.add_trace(go.Bar(x=[f"{int(k)}+ in a row" for k in marginal.streak_threshold],
                         y=marginal.qualify_rate_without * 100, name="without it",
                         marker_color=SERIES[1]))
    fig.update_layout(barmode="group", yaxis_title="Qualify (%)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Holding the win total fixed")
    fig = go.Figure()
    for colour, wins in zip(SERIES, (7, 8, 9)):
        grp = stratified[stratified.wins == wins]
        fig.add_trace(go.Scatter(x=grp.max_streak, y=grp.qualified_rate * 100,
                                 mode="lines+markers", name=f"{wins} wins",
                                 line=dict(width=3, color=colour), marker=dict(size=9)))
    fig.update_layout(xaxis_title="Longest win streak", yaxis_title="Qualify (%)",
                      yaxis_range=[0, 105], height=420)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Logistic regression on the same simulated seasons:**")
st.dataframe(
    logit.round(4)[["model", "term", "coef", "std_err", "p_value", "odds_ratio", "pseudo_r2"]],
    use_container_width=True, hide_index=True,
)
st.markdown(
    "Streak alone looks powerful. Put the win total in the model and its coefficient "
    "goes to zero and the fit does not improve. In a world built with no momentum at all, "
    "that is exactly the right answer — which is the point."
)

# ---------------------------------------------------------------- section 3
st.header("3. What the streak-capped model really says")
st.markdown(
    "Ask: *if a team never wins more than two in a row, how often does it reach 7 wins?* "
    "The old analysis answered this three incompatible ways."
)
table = _model_table()
show = table[["p_win", "v1_math_7plus", "v1_sim_7plus", "correct_conditional_7plus"]].copy()
show.columns = ["Win probability", "v1 “math model” (%)", "v1 “simulation” (%)", "Correct (%)"]
st.dataframe(show.round(2), use_container_width=True, hide_index=True)
st.markdown(
    f"""
- The **math constant (8.69%)** divided the qualifying count by all {2**14:,} sequences
  instead of by the {streaks.count_valid_sequences(14, 2):,} that satisfy the constraint.
  It answers "how likely is a team to be *both* streak-capped *and* on 7 wins", not the
  question that was asked.
- The **simulation** forced a loss after every second win. That is a different process:
  the forced loss costs a match but no coin flip, so it flatters the team.
- The **correct conditional model** — a p-coin season, conditioned on never winning
  three in a row — is the middle column's honest counterpart, computed exactly by DP.
"""
)

# ---------------------------------------------------------------- section 4
st.header("4. Try a season yourself")
p_win = st.slider("Per-match win probability", 0.25, 0.80, 0.50, 0.01)
cap = st.slider("Maximum consecutive wins allowed", 1, 5, 2)
c1, c2, c3 = st.columns(3)
c1.metric("P(7+ wins), unconstrained",
          f"{streaks.prob_at_least_wins(7, n_matches=14, p_win=p_win, model='unconstrained')*100:.1f}%")
c2.metric(f"P(7+ wins), capped at {cap}",
          f"{streaks.prob_at_least_wins(7, n_matches=14, p_win=p_win, model='conditional', cap=cap)*100:.1f}%")
c3.metric("P(3+ win streak), unconstrained",
          f"{streaks.prob_at_least_streak(3, n_matches=14, p_win=p_win, model='unconstrained')*100:.1f}%")

st.header("5. Test your own team's season")
st.markdown("Paste a result string — `W` win, `L` loss, `N` no result, `T` tie. Example: `WLWWLLWLWWLWLL`")
raw = st.text_input("Season results", value="WLWWLLWLWWLWLL").strip().upper()
if raw:
    seq = list(raw)
    wins = seq.count("W")
    st.write(
        f"**{len(seq)} matches · {wins} wins · longest streak {momentum.max_streak(seq)} · "
        f"{2*wins + seq.count('N') + seq.count('T')} points**"
    )
    z = momentum.runs_test_z(seq)
    if pd.isna(z):
        st.info("Runs test needs at least one win and one loss.")
    else:
        st.write(
            f"Wald–Wolfowitz runs z = **{z:+.2f}**. "
            + ("Slightly clumped, but well inside what chance produces."
               if abs(z) < 1.96 else
               "Unusually clumped for a season of this record." if z < 0 else
               "Unusually alternating for a season of this record.")
        )
        st.caption(
            "One season of 14 matches can almost never reach significance on its own — "
            "which is why the repository tests momentum by pooling every team-season."
        )
