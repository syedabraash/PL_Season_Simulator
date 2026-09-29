import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from schema import Fixture, ClubState
from simulate import run_simulations, SimulationConfig
from aggregate import summarize


def test_two_team_deterministic_win():
    """If Team A always beats Team B, A's final points should reflect that
    exactly (no draws/losses possible with p=1.0)."""
    fixtures = [
        Fixture(gameweek=1, home_team="A", away_team="B",
                p_home_win=1.0, p_draw=0.0, p_away_win=0.0)
    ]
    states = {
        "A": ClubState(name="A", points=10, played=5),
        "B": ClubState(name="B", points=10, played=5),
    }
    config = SimulationConfig(n_simulations=100, random_seed=1)
    results = run_simulations(fixtures, states, config)

    a_points = results[results["team"] == "A"]["final_points"]
    b_points = results[results["team"] == "B"]["final_points"]
    assert (a_points == 13).all()
    assert (b_points == 10).all()


def test_probabilities_sum_to_one_validation():
    """Fixture construction should reject badly miscalibrated inputs."""
    try:
        Fixture(gameweek=1, home_team="A", away_team="B",
                p_home_win=0.5, p_draw=0.5, p_away_win=0.5)
        assert False, "expected ValueError for probabilities summing to 1.5"
    except ValueError:
        pass


def test_summarize_probabilities_sum_reasonably():
    """Across all clubs, P(top4) summed should roughly equal 4 (4 slots),
    and P(relegation) summed should roughly equal 3 (3 slots), for a
    plausible simulation."""
    fixtures = [
        Fixture(gameweek=1, home_team=f"T{i}", away_team=f"T{i+1}",
                p_home_win=0.4, p_draw=0.3, p_away_win=0.3)
        for i in range(0, 19, 2)
    ]
    states = {f"T{i}": ClubState(name=f"T{i}", points=20, played=15) for i in range(20)}
    config = SimulationConfig(n_simulations=2000, random_seed=2)
    results = run_simulations(fixtures, states, config)
    summary = summarize(results)

    assert abs(summary["p_top4"].sum() - 4) < 0.5
    assert abs(summary["p_relegation"].sum() - 3) < 0.5


if __name__ == "__main__":
    test_two_team_deterministic_win()
    test_probabilities_sum_to_one_validation()
    test_summarize_probabilities_sum_reasonably()
    print("All tests passed.")
