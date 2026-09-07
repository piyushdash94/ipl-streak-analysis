"""Checks for the Cricsheet loader, run against synthetic fixtures written in
Cricsheet's exact JSON layout.

This matters because the download itself is blocked in some environments: the
parsing and aggregation logic still has to be verified, so that when the real
zip is available the output can be trusted rather than eyeballed.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cricsheet

TEAMS = [f"Team {c}" for c in "ABCDEFGH"]


def make_match(season_year, day, teams, winner=None, stage=None, match_number=None,
               result=None, eliminator=None, method=None):
    outcome = {}
    if winner is not None:
        outcome["winner"] = winner
        outcome["by"] = {"runs": 20}
    if result is not None:
        outcome["result"] = result
    if eliminator is not None:
        outcome["eliminator"] = eliminator
    if method is not None:
        outcome["method"] = method
    event = {"name": "Indian Premier League"}
    if stage is not None:
        event["stage"] = stage
    if match_number is not None:
        event["match_number"] = match_number
    return {
        "meta": {"data_version": "1.1.0"},
        "info": {
            "balls_per_over": 6,
            "dates": [f"{season_year}-04-{day:02d}"],
            "event": event,
            "gender": "male",
            "match_type": "T20",
            "outcome": outcome,
            "overs": 20,
            "season": str(season_year),
            "team_type": "club",
            "teams": list(teams),
            "venue": "Somewhere",
        },
    }


def build_season(season_year):
    """A full 8-team double round robin (56 league matches) plus 4 playoff matches."""
    matches, day, number = [], 1, 1
    for i, home in enumerate(TEAMS):
        for j, away in enumerate(TEAMS):
            if i == j:
                continue
            # Deterministic, lopsided result so Team A wins a lot and Team H little.
            winner = home if i < j else away
            matches.append(make_match(season_year, (day % 28) + 1, (home, away),
                                      winner=winner, match_number=number))
            day += 1
            number += 1
    top4 = TEAMS[:4]
    for stage, pair in [("Qualifier 1", (top4[0], top4[1])),
                        ("Eliminator", (top4[2], top4[3])),
                        ("Qualifier 2", (top4[1], top4[2])),
                        ("Final", (top4[0], top4[1]))]:
        matches.append(make_match(season_year, 28, pair, winner=pair[0], stage=stage))
    return matches


@pytest.fixture
def season_zip(tmp_path):
    path = tmp_path / "ipl_json.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for season in (2024, 2025):
            for k, match in enumerate(build_season(season)):
                archive.writestr(f"{season}{k:04d}.json", json.dumps(match))
    return path


def test_parse_matches_shapes(season_zip):
    matches = cricsheet.parse_matches(season_zip)
    assert len(matches) == 2 * 60
    assert set(matches["season"]) == {2024, 2025}
    assert matches.groupby("season")["is_league"].sum().tolist() == [56, 56]
    assert (~matches["is_league"]).sum() == 8


def test_build_team_seasons_is_consistent(season_zip):
    matches = cricsheet.parse_matches(season_zip)
    ts = cricsheet.build_team_seasons(matches)
    assert len(ts) == 16
    assert (ts["matches"] == 14).all()
    assert (ts["wins"] + ts["losses"] + ts["no_results"] == ts["matches"]).all()
    assert (ts["points"] == 2 * ts["wins"] + ts["no_results"]).all()
    assert (ts["sequence"].str.len() == ts["matches"]).all()
    for season, grp in ts.groupby("season"):
        assert grp["points"].sum() == 2 * 56, season
        assert grp["qualified"].sum() == 4, season


def test_validation_report_is_clean(season_zip):
    matches = cricsheet.parse_matches(season_zip)
    report = cricsheet.validate(cricsheet.build_team_seasons(matches), matches)
    assert report["all_ok"].all(), report


def test_validation_flags_a_broken_season(season_zip):
    """Drop a league match: the report must notice, rather than quietly average over it."""
    matches = cricsheet.parse_matches(season_zip)
    broken = matches.drop(matches[matches["is_league"] & (matches["season"] == 2024)].index[0])
    report = cricsheet.validate(cricsheet.build_team_seasons(broken), broken)
    row = report[report["season"] == 2024].iloc[0]
    assert not row["balanced_schedule"]
    assert not row["all_ok"]
    assert report[report["season"] == 2025].iloc[0]["all_ok"]


def test_no_result_and_tie_handling(tmp_path):
    matches = [
        make_match(2025, 1, ("Team A", "Team B"), result="no result", match_number=1),
        make_match(2025, 2, ("Team A", "Team B"), result="tie", eliminator="Team A", match_number=2),
        make_match(2025, 3, ("Team A", "Team B"), result="tie", match_number=3),
        make_match(2025, 4, ("Team A", "Team B"), winner="Team A", method="D/L", match_number=4),
    ]
    path = tmp_path / "small.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for k, m in enumerate(matches):
            archive.writestr(f"{k}.json", json.dumps(m))
    ts = cricsheet.build_team_seasons(cricsheet.parse_matches(path)).set_index("team")
    assert ts.loc["Team A", "sequence"] == "NWTW"
    assert ts.loc["Team B", "sequence"] == "NLTL"
    assert ts.loc["Team A", "points"] == 2 * 2 + 2   # two wins, a no-result, a tie
    assert ts.loc["Team B", "points"] == 2
    assert ts.loc["Team A", "max_streak"] == 1       # the tie breaks the run
    assert ts.loc["Team A", "points"] + ts.loc["Team B", "points"] == 8


def test_season_year_comes_from_dates_not_the_season_label(tmp_path):
    """Cricsheet labels IPL 2020 as season '2020/21'; the dates are the truth."""
    match = make_match(2020, 20, ("Team A", "Team B"), winner="Team A", match_number=1)
    match["info"]["season"] = "2020/21"
    path = tmp_path / "s.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("m.json", json.dumps(match))
    assert cricsheet.parse_matches(path)["season"].tolist() == [2020]


def test_directory_source_works(tmp_path):
    for k, m in enumerate(build_season(2025)):
        (tmp_path / f"{k}.json").write_text(json.dumps(m))
    assert len(cricsheet.parse_matches(tmp_path)) == 60


def test_renamed_franchises_are_unified(tmp_path):
    m = make_match(2024, 1, ("Royal Challengers Bangalore", "Kings XI Punjab"),
                   winner="Kings XI Punjab", match_number=1)
    path = tmp_path / "r.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("m.json", json.dumps(m))
    row = cricsheet.parse_matches(path).iloc[0]
    assert row["team_a"] == "Royal Challengers Bengaluru"
    assert row["winner"] == "Punjab Kings"


def test_empty_source_raises(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "nothing here")
    with pytest.raises(ValueError, match="no matches parsed"):
        cricsheet.parse_matches(path)


def test_download_failure_is_explicit(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("blocked by egress policy")

    monkeypatch.setattr(cricsheet.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="could not download"):
        cricsheet.download(tmp_path / "x.zip")
