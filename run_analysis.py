#!/usr/bin/env python3
"""Reproduce every number and figure in the write-up.

    python run_analysis.py                 # simulation-based analysis
    python run_analysis.py --with-data     # also run the Cricsheet analysis

Writes reports/results.json and reports/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from src import analysis, cricsheet, viz

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true",
                        help="download/parse Cricsheet IPL results and analyse them")
    parser.add_argument("--data", default=None,
                        help="path to an already-downloaded ipl_json.zip or a directory of JSON")
    parser.add_argument("--seasons", type=int, default=30_000,
                        help="simulated seasons for the qualification curves")
    parser.add_argument("--confound-seasons", type=int, default=40_000,
                        help="simulated seasons for the streak/wins confound analysis")
    parser.add_argument("--skip-power", action="store_true",
                        help="skip the (slow) momentum power curve")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    print("1/5  exact model comparison")
    models = analysis.model_comparison()
    results["model_comparison"] = models
    results["sequence_counts"] = {
        "all_sequences": models.attrs["n_all_sequences"],
        "valid_sequences_max_2_streak": models.attrs["n_valid_sequences"],
        "v1_denominator_reproduced": round(models.attrs["v1_denominator_check"], 2),
    }
    viz.plot_model_comparison(models, FIGURES / "model_comparison.png")

    print("2/5  streak frequency")
    freq = analysis.streak_frequency()
    results["streak_frequency"] = freq
    viz.plot_streak_frequency(freq, FIGURES / "streak_frequency.png")

    print("3/5  league format comparison")
    curves, cutoffs = analysis.format_comparison(n_seasons=args.seasons)
    results["qualification_curve"] = curves
    results["cutoff_distribution"] = cutoffs
    viz.plot_qualification_curve(curves, FIGURES / "qualification_curve.png")

    print("4/5  streak-vs-wins confound")
    confound = analysis.confound_analysis(n_seasons=args.confound_seasons)
    results["confound_marginal"] = confound["marginal"]
    results["confound_stratified"] = confound["stratified"]
    results["confound_logit"] = confound["logit"]
    viz.plot_confound(confound["marginal"], confound["stratified"], FIGURES / "confound.png")

    if args.skip_power:
        print("5/5  power curve  (skipped)")
    else:
        print("5/5  momentum power curve")
        results["momentum_power"] = analysis.momentum_power_curve()

    if args.with_data or args.data:
        print("     Cricsheet analysis")
        try:
            matches, team_seasons, report = cricsheet.load(args.data)
        except (RuntimeError, ValueError) as exc:
            print(f"     SKIPPED: {exc}")
            results["real_data"] = {"available": False, "reason": str(exc)}
        else:
            team_seasons.to_csv(REPORTS / "team_seasons.csv", index=False)
            report.to_csv(REPORTS / "data_validation.csv", index=False)
            print(report.to_string(index=False))
            if not report["all_ok"].all():
                print("     WARNING: validation flagged seasons above — inspect before trusting.")
            real = analysis.real_data_analysis(team_seasons)
            results["real_data"] = {
                "available": True,
                "seasons": sorted(team_seasons["season"].unique().tolist()),
                "team_seasons": len(team_seasons),
                "validation": report,
                **{k: v for k, v in real.items()},
            }

    with (REPORTS / "results.json").open("w") as handle:
        json.dump(results, handle, indent=2, default=_jsonable)
    print(f"\nWrote {REPORTS/'results.json'} and {len(list(FIGURES.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
