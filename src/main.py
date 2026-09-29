import pandas as pd

from schema import fixtures_from_dataframe, club_states_from_dataframe
from calibration import full_reliability_report, print_calibration_warning_if_needed
from simulate import run_simulations, SimulationConfig
from aggregate import summarize, position_distribution
from synthetic_data import (
    generate_remaining_fixtures,
    generate_current_table,
    generate_backtest_predictions,
)


def main():
    print("=" * 60)
    print("STEP 1: Calibration check on backtest predictions")
    print("=" * 60)
    backtest_df = generate_backtest_predictions()
    report = full_reliability_report(backtest_df)
    print_calibration_warning_if_needed(report)
    report.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\calibration_report.csv", index=False)
    print("-> outputs/calibration_report.csv\n")

    print("=" * 60)
    print("STEP 2: Load remaining fixtures + current table")
    print("=" * 60)
    fixtures_df = generate_remaining_fixtures()
    table_df = generate_current_table()
    fixtures = fixtures_from_dataframe(fixtures_df)
    states = club_states_from_dataframe(table_df)
    print(f"{len(fixtures)} remaining fixtures across "
          f"{fixtures_df['gameweek'].nunique()} gameweeks, "
          f"{len(states)} clubs.\n")

    print("=" * 60)
    print("STEP 3: Running Monte Carlo simulation")
    print("=" * 60)
    config = SimulationConfig(n_simulations=10_000, random_seed=42)
    sim_results = run_simulations(fixtures, states, config)
    print(f"Ran {config.n_simulations:,} simulations.\n")

    print("=" * 60)
    print("STEP 4: Aggregating results")
    print("=" * 60)
    summary = summarize(sim_results)
    print(summary.to_string())
    summary.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\season_probabilities.csv")

    pos_dist = position_distribution(sim_results)
    pos_dist.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\position_distribution.csv")

    print("\n-> outputs/season_probabilities.csv")
    print("-> outputs/position_distribution.csv")


if __name__ == "__main__":
    main()
