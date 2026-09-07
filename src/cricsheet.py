"""Build IPL team-season tables from Cricsheet ball-by-ball JSON.

Cricsheet (https://cricsheet.org, CC BY 4.0) publishes every IPL match as a
JSON file.  ``ipl_json.zip`` is a few MB and covers every season, which makes it
the right source for this project: it is complete, versioned, and machine
readable, unlike the hand-typed W/L strings v1 relied on.

The one design decision worth flagging: **qualification is read off the data,
not inferred from points.**  A team qualified for the playoffs in season Y if
and only if it appears in a playoff match that season.  That avoids having to
reconstruct Net Run Rate from deliveries -- which is genuinely fiddly once
Duckworth-Lewis revisions and all-out-before-the-full-quota rules are involved,
and which would be an unnecessary source of error for the questions asked here.

Network note: this module has to reach cricsheet.org.  Some sandboxed
environments block it; :func:`download` then raises with a clear message rather
than silently producing an empty table.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

CRICSHEET_IPL_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"

#: Franchises that have been renamed.  Cricsheet uses the name current at the
#: time of the match, so cross-season work needs a canonical label.
CANONICAL_TEAM_NAMES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Gujarat Lions": "Gujarat Lions",
    "Pune Warriors": "Pune Warriors India",
}

#: Cricsheet ``info.event.stage`` values are absent for league matches.  Any
#: stage at all means the match was part of the knockout phase.
LEAGUE_STAGE = None


def canonical_team(name: str) -> str:
    return CANONICAL_TEAM_NAMES.get(name, name)


def download(dest: str | Path = "data/ipl_json.zip", force: bool = False) -> Path:
    """Fetch ``ipl_json.zip`` from Cricsheet, caching to ``dest``."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(CRICSHEET_IPL_JSON_URL, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"could not download {CRICSHEET_IPL_JSON_URL} ({exc}). "
            "Run this where cricsheet.org is reachable, or download the zip "
            f"by hand and place it at {dest}."
        ) from exc
    dest.write_bytes(payload)
    return dest


def iter_match_info(source: str | Path) -> Iterator[dict]:
    """Yield the ``info`` block of every match in a Cricsheet zip or directory."""
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.glob("*.json")):
            with path.open() as handle:
                yield json.load(handle)["info"]
        return
    with zipfile.ZipFile(source) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json"):
                continue
            with archive.open(name) as handle:
                yield json.load(handle)["info"]


def _season_year(info: dict) -> int:
    """Calendar year of the match.  Every IPL season sits inside one year.

    Cricsheet's own ``season`` field is inconsistent ("2007/08", "2020/21",
    "2023"), so the dates are the reliable source.
    """
    dates = info["dates"]
    return max(int(str(d)[:4]) for d in dates)


def parse_matches(source: str | Path) -> pd.DataFrame:
    """One row per match: teams, result, stage and season."""
    rows = []
    for info in iter_match_info(source):
        teams = [canonical_team(t) for t in info["teams"]]
        if len(teams) != 2:
            continue
        outcome = info.get("outcome", {})
        winner = outcome.get("winner") or outcome.get("eliminator")
        event = info.get("event") or {}
        rows.append(
            {
                "season": _season_year(info),
                "date": min(str(d) for d in info["dates"]),
                "match_number": event.get("match_number"),
                "stage": event.get("stage"),
                "team_a": teams[0],
                "team_b": teams[1],
                "winner": canonical_team(winner) if winner else None,
                "result": outcome.get("result"),  # 'no result' / 'tie' / None
                "method": outcome.get("method"),  # 'D/L' where applicable
            }
        )
    if not rows:
        raise ValueError(f"no matches parsed from {source}")
    df = pd.DataFrame(rows)
    df["is_league"] = df["stage"].isna()
    return df.sort_values(["season", "date", "match_number"], na_position="last").reset_index(
        drop=True
    )


def build_team_seasons(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per season, with the ordered league-match result string.

    Columns: ``season, team, sequence, wins, losses, no_results, points,
    max_streak, matches, qualified``.  ``sequence`` is chronological, which is
    what every streak statistic in this project depends on.
    """
    from .momentum import max_streak as _max_streak

    league = matches[matches["is_league"]]
    playoff_teams = defaultdict(set)
    for _, row in matches[~matches["is_league"]].iterrows():
        playoff_teams[row["season"]].update({row["team_a"], row["team_b"]})

    records: dict[tuple[int, str], list[str]] = defaultdict(list)
    for _, row in league.iterrows():
        for team in (row["team_a"], row["team_b"]):
            if row["result"] == "no result":
                records[(row["season"], team)].append("NR")
            elif pd.isna(row["winner"]):
                records[(row["season"], team)].append("T")  # tie, no eliminator
            else:
                records[(row["season"], team)].append("W" if row["winner"] == team else "L")

    rows = []
    for (season, team), sequence in records.items():
        wins = sequence.count("W")
        drawn = sequence.count("NR") + sequence.count("T")
        rows.append(
            {
                "season": season,
                "team": team,
                "sequence": "".join(
                    {"W": "W", "L": "L", "NR": "N", "T": "T"}[r] for r in sequence
                ),
                "matches": len(sequence),
                "wins": wins,
                "losses": sequence.count("L"),
                "no_results": drawn,
                "points": 2 * wins + drawn,
                "max_streak": _max_streak(sequence),
                "qualified": team in playoff_teams.get(season, set()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["season", "points", "wins"], ascending=[True, False, False])
    return out.reset_index(drop=True)


def validate(team_seasons: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per-season sanity report.  Read this before trusting any downstream number.

    Flags seasons where teams played different numbers of league matches, where
    the points in the table do not equal 2 per league match, or where the number
    of playoff qualifiers is not 4 (top-4 has been the format since 2008).
    """
    rows = []
    for season, grp in team_seasons.groupby("season"):
        league_matches = int((matches["is_league"] & (matches["season"] == season)).sum())
        counts = sorted(grp["matches"].unique().tolist())
        rows.append(
            {
                "season": season,
                "teams": len(grp),
                "league_matches": league_matches,
                "matches_per_team": counts,
                "balanced_schedule": len(counts) == 1,
                "points_conserved": int(grp["points"].sum()) == 2 * league_matches,
                "qualifiers": int(grp["qualified"].sum()),
                "qualifiers_ok": int(grp["qualified"].sum()) == 4,
                "cutoff_points": int(grp[grp["qualified"]]["points"].min()) if grp["qualified"].any() else None,
                "highest_missing_points": int(grp[~grp["qualified"]]["points"].max())
                if (~grp["qualified"]).any()
                else None,
            }
        )
    report = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    report["all_ok"] = report["balanced_schedule"] & report["points_conserved"] & report["qualifiers_ok"]
    return report


def load(source: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: download if needed, then return (matches, team_seasons, report)."""
    source = Path(source) if source is not None else download()
    matches = parse_matches(source)
    team_seasons = build_team_seasons(matches)
    return matches, team_seasons, validate(team_seasons, matches)
