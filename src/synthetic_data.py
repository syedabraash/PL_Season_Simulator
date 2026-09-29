import numpy as np
import pandas as pd

CLUBS_20 = [
    "Arsenal", "Man City", "Liverpool", "Chelsea", "Newcastle",
    "Aston Villa", "Tottenham", "Man United", "Brighton", "Bournemouth",
    "Crystal Palace", "Fulham", "Brentford", "Nottm Forest", "West Ham",
    "Everton", "Wolves", "Burnley", "Coventry City", "Hull City",
]

# Rough relative strength (higher = stronger). Arbitrary, for demo only.
_STRENGTH = {
    "Man City": 92, "Arsenal": 90, "Liverpool": 88, "Chelsea": 82,
    "Newcastle": 78, "Aston Villa": 76, "Tottenham": 75, "Brighton": 72,
    "Man United": 71, "West Ham": 68, "Crystal Palace": 66, "Fulham": 65,
    "Brentford": 64, "Bournemouth": 63, "Nottm Forest": 62, "Wolves": 58,
    "Everton": 57, "Burnley": 50, "Coventry City": 45, "Hull City": 43,
}

HOME_ADVANTAGE = 6  # flat strength bump for home side, simple demo heuristic


def _match_probabilities(home: str, away: str) -> tuple[float, float, float]:
    diff = (_STRENGTH[home] + HOME_ADVANTAGE) - _STRENGTH[away]
    p_home = 1 / (1 + np.exp(-diff / 12))
    p_away = 1 / (1 + np.exp(diff / 12))
    # carve out draw probability, then renormalize
    p_draw = 0.26 - 0.05 * abs(diff) / 20
    p_draw = max(0.15, min(0.30, p_draw))
    remaining = 1 - p_draw
    p_home_final = p_home / (p_home + p_away) * remaining
    p_away_final = p_away / (p_home + p_away) * remaining
    return round(p_home_final, 4), round(p_draw, 4), round(p_away_final, 4)


def generate_remaining_fixtures(
    clubs: list[str] = CLUBS_20, remaining_gameweeks: int = 10, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, remaining_gameweeks + 1):
        shuffled = clubs.copy()
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled), 2):
            home, away = shuffled[i], shuffled[i + 1]
            p_h, p_d, p_a = _match_probabilities(home, away)
            rows.append({
                "gameweek": gw, "home_team": home, "away_team": away,
                "p_home_win": p_h, "p_draw": p_d, "p_away_win": p_a,
            })
    return pd.DataFrame(rows)


def generate_current_table(clubs: list[str] = CLUBS_20, played: int = 28, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for club in clubs:
        # Points-per-game roughly tracks strength, plus noise
        strength = _STRENGTH[club]
        ppg = 0.4 + (strength - 43) / (92 - 43) * 2.0  # maps ~43-92 -> ~0.4-2.4
        points = int(round(ppg * played + rng.normal(0, 3)))
        points = max(0, points)
        gd = int(round((strength - 65) / 4 + rng.normal(0, 3)))
        rows.append({"team": club, "points": points, "played": played, "goal_difference": gd})
    return pd.DataFrame(rows)


def generate_backtest_predictions(n_matches: int = 800, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_matches):
        home = rng.choice(CLUBS_20)
        away = rng.choice([c for c in CLUBS_20 if c != home])
        p_h, p_d, p_a = _match_probabilities(home, away)
        actual = rng.choice(["H", "D", "A"], p=[p_h, p_d, p_a])
        rows.append({
            "home_team": home, "away_team": away,
            "p_home_win": p_h, "p_draw": p_d, "p_away_win": p_a,
            "actual_result": actual,
        })
    return pd.DataFrame(rows)
