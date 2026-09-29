import numpy as np
import pandas as pd

EWM_SPAN = 8  # ~half a season's worth of recency weighting, matches typical "form" windows


def _team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "goals_for", "away_goals": "goals_against",
        "home_possession": "possession", "home_shots": "shots", "home_sot": "sot",
    }).copy()
    home["venue"] = "H"
    home["points"] = home["result"].map({"H": 3, "D": 1, "A": 0})

    away = matches.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "goals_for", "home_goals": "goals_against",
        "away_possession": "possession", "away_shots": "shots", "away_sot": "sot",
    }).copy()
    away["venue"] = "A"
    away["points"] = away["result"].map({"H": 0, "D": 1, "A": 3})

    keep = ["date", "season", "team", "opponent", "venue", "goals_for",
            "goals_against", "possession", "shots", "sot", "points"]
    log = pd.concat([home[keep], away[keep]], ignore_index=True)
    return log.sort_values(["team", "date"]).reset_index(drop=True)


def build_team_form_features(matches: pd.DataFrame, ewm_span: int = EWM_SPAN) -> pd.DataFrame:
    log = _team_match_log(matches)
    log["goal_difference"] = log["goals_for"] - log["goals_against"]

    feature_cols = ["points", "goals_for", "goals_against", "goal_difference",
                     "possession", "shots"]
    out_frames = []

    for team, team_log in log.groupby("team"):
        team_log = team_log.sort_values("date").reset_index(drop=True)
        season_frames = []
        prior_season_end_state = None

        for season, season_log in team_log.groupby("season", sort=True):
            season_log = season_log.reset_index(drop=True)
            ewm_vals = {}
            for col in feature_cols:
                series = season_log[col].astype(float)
                if prior_season_end_state is not None and col in prior_season_end_state \
                        and not np.isnan(prior_season_end_state[col]):
                    seed = prior_season_end_state[col] * 0.5
                    series_with_seed = pd.concat([pd.Series([seed]), series], ignore_index=True)
                    ewm = series_with_seed.ewm(span=ewm_span, adjust=False).mean().iloc[1:].reset_index(drop=True)
                else:
                    ewm = series.ewm(span=ewm_span, adjust=False).mean()
                # Shift by 1: features available BEFORE match i use only
                # matches < i, never match i's own result (avoids leakage).
                ewm_vals[f"form_{col}_ewm"] = ewm.shift(1)

            season_features = season_log[["date", "season", "team", "opponent", "venue"]].copy()
            for k, v in ewm_vals.items():
                season_features[k] = v
            season_frames.append(season_features)

            # Record end-of-season state (full EWM including final match)
            # to seed next season's opening form.
            prior_season_end_state = {}
            for col in feature_cols:
                series = season_log[col].astype(float)
                ewm_full = series.ewm(span=ewm_span, adjust=False).mean()
                prior_season_end_state[col] = ewm_full.iloc[-1] if len(ewm_full) else np.nan

        out_frames.append(pd.concat(season_frames, ignore_index=True))

    return pd.concat(out_frames, ignore_index=True)


def build_fixture_training_table(matches: pd.DataFrame, ewm_span: int = EWM_SPAN):
    form = build_team_form_features(matches, ewm_span)

    home_form = form.rename(columns={
        "team": "home_team", "opponent": "home_opponent", "venue": "home_venue"
    })
    home_form = home_form.rename(columns={c: f"home_{c}" for c in home_form.columns if c.startswith("form_")})

    away_form = form.rename(columns={
        "team": "away_team", "opponent": "away_opponent", "venue": "away_venue"
    })
    away_form = away_form.rename(columns={c: f"away_{c}" for c in away_form.columns if c.startswith("form_")})

    merged = matches.merge(
        home_form.drop(columns=["home_opponent", "home_venue"]),
        on=["date", "season", "home_team"], how="left"
    ).merge(
        away_form.drop(columns=["away_opponent", "away_venue"]),
        on=["date", "season", "away_team"], how="left"
    )

    feature_cols = [c for c in merged.columns if c.startswith("home_form_") or c.startswith("away_form_")]
    before = len(merged)
    merged = merged.dropna(subset=feature_cols)
    dropped = before - len(merged)
    if dropped:
        print(f"Dropped {dropped}/{before} matches with no prior-form history "
              f"(each team's opening matches in the dataset).")

    return merged, feature_cols
