import numpy as np
import pandas as pd

from features import _team_match_log, EWM_SPAN
from data_loader import CLUBS_WITHOUT_HISTORY, CLUB_NAME_MAP

# Only these clubs have a full own-perspective sheet in PL_Club_5years.xlsx.
# Any other name that shows up as an "away_team" opponent (e.g. Wolves,
# Burnley, West Ham -- clubs without their own sheet in the historical
# file) has partial, opponent-only records with NaN possession/shots, and
# must be excluded from strength/seed calculations or it corrupts them.
KNOWN_CLUBS = list(CLUB_NAME_MAP.values())

FEATURE_COLS = ["points", "goals_for", "goals_against", "goal_difference",
                 "possession", "shots"]
SEASON_SHRINK = 0.5


def _end_of_season_ewm(matches: pd.DataFrame, season: str, ewm_span: int = EWM_SPAN) -> pd.DataFrame:
    log = _team_match_log(matches)
    log["goal_difference"] = log["goals_for"] - log["goals_against"]
    season_log = log[(log["season"] == season) & (log["team"].isin(KNOWN_CLUBS))]

    rows = []
    for team, team_log in season_log.groupby("team"):
        team_log = team_log.sort_values("date")
        row = {"team": team}
        for col in FEATURE_COLS:
            series = team_log[col].astype(float)
            ewm = series.ewm(span=ewm_span, adjust=False).mean()
            row[col] = ewm.iloc[-1] if len(ewm) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("team")


def _apply_n_updates(seed: dict, per_game_avg: dict, n: int, ewm_span: int = EWM_SPAN) -> dict:
    alpha = 2 / (ewm_span + 1)
    state = dict(seed)
    for col in ["points", "goals_for", "goals_against", "goal_difference"]:
        value = state[col]
        for _ in range(n):
            value = alpha * per_game_avg[col] + (1 - alpha) * value
        state[col] = value
    # possession/shots: no current-season data available, carried forward unchanged
    return state


def build_current_strength(
    historical_matches: pd.DataFrame,
    current_table: pd.DataFrame,
    all_20_clubs: list[str],
    prior_season: str = "2025-26",
    ewm_span: int = EWM_SPAN,
) -> pd.DataFrame:
    end_of_season = _end_of_season_ewm(historical_matches, prior_season, ewm_span)

    # Bottom-3 by points at end of prior season = proxy prior for newly
    # promoted clubs with no history at all.
    last_season_points = (
        historical_matches[historical_matches["season"] == prior_season]
    )
    # Reconstruct a simple final points tally from the historical match log
    # (3/1/0 per result) to rank clubs and find the bottom 3.
    from features import _team_match_log as _tml
    log = _tml(historical_matches)
    log = log[(log["season"] == prior_season) & (log["team"].isin(KNOWN_CLUBS))]
    final_points = log.groupby("team")["points"].sum().sort_values()
    bottom_3 = final_points.index[:3].tolist()
    promoted_prior_seed = end_of_season.loc[end_of_season.index.intersection(bottom_3)].mean()

    current_table = current_table.set_index("team")
    rows = {}
    for team in all_20_clubs:
        n_games = int(current_table.loc[team, "played"]) if team in current_table.index else 4
        per_game_avg = {
            "points": current_table.loc[team, "points"] / n_games if team in current_table.index else np.nan,
            "goals_for": current_table.loc[team, "goals_for"] / n_games if team in current_table.index else np.nan,
            "goals_against": current_table.loc[team, "goals_against"] / n_games if team in current_table.index else np.nan,
            "goal_difference": current_table.loc[team, "goal_difference"] / n_games if team in current_table.index else np.nan,
        }

        if team in CLUBS_WITHOUT_HISTORY:
            seed_series = promoted_prior_seed
        elif team in end_of_season.index:
            seed_series = end_of_season.loc[team]
        else:
            # Team in the 20-club list but genuinely missing from both
            # history and the proxy -- fall back to full-league average.
            seed_series = end_of_season.mean()

        seed = {col: seed_series[col] * SEASON_SHRINK for col in FEATURE_COLS}
        updated = _apply_n_updates(seed, per_game_avg, n_games, ewm_span)
        rows[team] = updated

    return pd.DataFrame(rows).T[FEATURE_COLS]


def strength_to_fixture_features(strength: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    home = strength.add_prefix("home_form_").add_suffix("_ewm")
    away = strength.add_prefix("away_form_").add_suffix("_ewm")

    out = fixtures_df.merge(home, left_on="home_team", right_index=True, how="left")
    out = out.merge(away, left_on="away_team", right_index=True, how="left")
    return out
