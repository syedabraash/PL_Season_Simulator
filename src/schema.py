from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Fixture:
    """One remaining fixture with model-predicted outcome probabilities."""
    gameweek: int
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float

    def __post_init__(self):
        total = self.p_home_win + self.p_draw + self.p_away_win
        if not (0.98 <= total <= 1.02):
            raise ValueError(
                f"{self.home_team} vs {self.away_team}: probabilities sum to "
                f"{total:.4f}, expected ~1.0. Check model output / calibration."
            )


@dataclass
class ClubState:
    name: str
    points: int
    played: int
    goal_difference: int = 0  # used only for tiebreaks; optional in Option A


def fixtures_from_dataframe(df: pd.DataFrame) -> list[Fixture]:
    required = {"gameweek", "home_team", "away_team",
                "p_home_win", "p_draw", "p_away_win"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fixtures dataframe missing columns: {missing}")

    return [
        Fixture(
            gameweek=int(row.gameweek),
            home_team=row.home_team,
            away_team=row.away_team,
            p_home_win=float(row.p_home_win),
            p_draw=float(row.p_draw),
            p_away_win=float(row.p_away_win),
        )
        for row in df.itertuples(index=False)
    ]


def club_states_from_dataframe(df: pd.DataFrame) -> dict[str, ClubState]:
    has_gd = "goal_difference" in df.columns
    return {
        row.team: ClubState(
            name=row.team,
            points=int(row.points),
            played=int(row.played),
            goal_difference=int(row.goal_difference) if has_gd else 0,
        )
        for row in df.itertuples(index=False)
    }
