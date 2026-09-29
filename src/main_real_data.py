import pandas as pd

from data_loader import (
    load_historical_matches, load_current_season_table, CLUBS_2026_27
)
from features import build_fixture_training_table
from train_model import train_and_evaluate, predict_fixture_probabilities
from calibration import full_reliability_report, print_calibration_warning_if_needed
from current_strength import build_current_strength, strength_to_fixture_features
from fixture_generator import generate_remaining_fixtures_df
from schema import fixtures_from_dataframe, club_states_from_dataframe
from simulate import run_simulations, SimulationConfig
from aggregate import summarize, position_distribution

CLUB_5YEARS_PATH = "../data/PL_Club_5years.xlsx"
STATS_2627_PATH = "../data/PL_26_27_Stats.xlsx"


def main():
    print("=" * 70)
    print("STEP 1: Load + unify 5 seasons of historical match data")
    print("=" * 70)
    matches = load_historical_matches(CLUB_5YEARS_PATH)
    print(f"{len(matches)} unified historical fixtures, "
          f"{matches['season'].nunique()} seasons.\n")

    print("=" * 70)
    print("STEP 2: Engineer rolling form features + build training table")
    print("=" * 70)
    train_df, feature_cols = build_fixture_training_table(matches)
    print(f"{len(train_df)} training rows with complete features.\n")

    print("=" * 70)
    print("STEP 3: Train + evaluate model (season-based train/test split)")
    print("=" * 70)
    model, scaler, model_name, needs_scaling, acc, logloss, test_preds = \
        train_and_evaluate(train_df, feature_cols)
    print()

    print("=" * 70)
    print("STEP 4: Calibration check on held-out 2025-26 season")
    print("=" * 70)
    report = full_reliability_report(test_preds)
    print_calibration_warning_if_needed(report)
    report.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\calibration_report_real.csv", index=False)
    print("-> outputs/calibration_report_real.csv\n")

    print("=" * 70)
    print("STEP 5: Build current (2026-27) team strength")
    print("=" * 70)
    current_table = load_current_season_table(STATS_2627_PATH, STATS_2627_PATH)
    strength = build_current_strength(matches, current_table, CLUBS_2026_27)
    print(strength.round(2).to_string())
    print()

    print("=" * 70)
    print("STEP 6: Generate remaining fixtures (round-robin -- see "
          "fixture_generator.py for why)")
    print("=" * 70)
    remaining_fixtures_raw = generate_remaining_fixtures_df(CLUBS_2026_27, already_played_rounds=4)
    fixtures_with_features = strength_to_fixture_features(strength, remaining_fixtures_raw)
    print(f"{len(fixtures_with_features)} remaining fixtures generated.\n")

    print("=" * 70)
    print("STEP 7: Predict fixture probabilities with the trained model")
    print("=" * 70)
    fixture_probs = predict_fixture_probabilities(
        model, scaler, needs_scaling, fixtures_with_features, feature_cols
    )
    fixture_probs.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\remaining_fixture_probabilities.csv", index=False)
    print("-> outputs/remaining_fixture_probabilities.csv\n")

    print("=" * 70)
    print("STEP 8: Monte Carlo simulation")
    print("=" * 70)
    fixtures = fixtures_from_dataframe(fixture_probs)
    states = club_states_from_dataframe(
        current_table.rename(columns={"goal_difference": "goal_difference"})
    )
    config = SimulationConfig(n_simulations=10_000, random_seed=42)
    sim_results = run_simulations(fixtures, states, config)
    print(f"Ran {config.n_simulations:,} simulations.\n")

    print("=" * 70)
    print("STEP 9: Aggregate results")
    print("=" * 70)
    summary = summarize(sim_results)
    print(summary.to_string())
    summary.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\season_probabilities_real.csv")

    pos_dist = position_distribution(sim_results)
    pos_dist.to_csv("C:\\Users\\syedt\\Desktop\\PL_Season_Simulator\\outputs\\position_distribution_real.csv")

    print("\n-> outputs/season_probabilities_real.csv")
    print("-> outputs/position_distribution_real.csv")


if __name__ == "__main__":
    main()
