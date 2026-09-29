from dataclasses import dataclass
import numpy as np
import pandas as pd

from schema import Fixture, ClubState


@dataclass
class SimulationConfig:
    n_simulations: int = 10_000
    random_seed: int | None = 42
    tiebreak: str = "random"  # "random" or "current_gd" (needs goal_difference in ClubState)


def _draw_outcomes_vectorized(
    fixtures: list[Fixture], n_simulations: int, rng: np.random.Generator
) -> np.ndarray:
    n_fixtures = len(fixtures)
    probs = np.array([[f.p_home_win, f.p_draw, f.p_away_win] for f in fixtures])
    # Normalize defensively in case of tiny floating point drift
    probs = probs / probs.sum(axis=1, keepdims=True)

    outcomes = np.empty((n_simulations, n_fixtures), dtype=np.int8)
    for j in range(n_fixtures):
        outcomes[:, j] = rng.choice(3, size=n_simulations, p=probs[j])
    return outcomes


def run_simulations(
    fixtures: list[Fixture],
    current_states: dict[str, ClubState],
    config: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    teams = sorted(current_states.keys())
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    outcomes = _draw_outcomes_vectorized(fixtures, config.n_simulations, rng)

    # Start every simulation from the real current points total
    base_points = np.array([current_states[t].points for t in teams])
    points = np.tile(base_points, (config.n_simulations, 1)).astype(np.int32)

    for j, fixture in enumerate(fixtures):
        home_i = team_idx[fixture.home_team]
        away_i = team_idx[fixture.away_team]
        result = outcomes[:, j]  # 0=home win, 1=draw, 2=away win

        home_win_mask = result == 0
        draw_mask = result == 1
        away_win_mask = result == 2

        points[home_win_mask, home_i] += 3
        points[away_win_mask, away_i] += 3
        points[draw_mask, home_i] += 1
        points[draw_mask, away_i] += 1

    # Tiebreak jitter: tiny random tiebreaker so ranking is well-defined.
    # This does NOT affect points shown to the user -- it only resolves
    # ordering when two clubs finish level. For accurate GD-based
    # tiebreaks, upgrade to Option B (scoreline simulation).
    if config.tiebreak == "random":
        tiebreak_noise = rng.random((config.n_simulations, n_teams)) * 1e-6
    else:
        base_gd = np.array([current_states[t].goal_difference for t in teams])
        tiebreak_noise = np.tile(base_gd, (config.n_simulations, 1)) * 1e-6

    ranking_key = points + tiebreak_noise
    # argsort descending -> position 1 = best
    positions = (-ranking_key).argsort(axis=1).argsort(axis=1) + 1

    sim_ids = np.repeat(np.arange(config.n_simulations), n_teams)
    team_col = np.tile(teams, config.n_simulations)
    points_col = points.flatten()
    positions_col = positions.flatten()

    return pd.DataFrame({
        "sim_id": sim_ids,
        "team": team_col,
        "final_points": points_col,
        "final_position": positions_col,
    })
