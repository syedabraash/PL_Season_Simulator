import numpy as np
import pandas as pd


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean(np.sum((y_proba - y_true) ** 2, axis=1)))


def reliability_table(
    y_true_outcome: pd.Series,
    p_predicted: pd.Series,
    outcome_label: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true_outcome.astype(int), "p": p_predicted})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1),
                        include_lowest=True)

    grouped = df.groupby("bin", observed=True).agg(
        n_matches=("y", "size"),
        predicted_mean=("p", "mean"),
        observed_rate=("y", "mean"),
    ).reset_index()
    grouped["outcome"] = outcome_label
    grouped["gap"] = grouped["predicted_mean"] - grouped["observed_rate"]
    return grouped


def full_reliability_report(backtest_df: pd.DataFrame) -> pd.DataFrame:
    reports = []
    for label, col, code in [
        ("home_win", "p_home_win", "H"),
        ("draw", "p_draw", "D"),
        ("away_win", "p_away_win", "A"),
    ]:
        y_true = backtest_df["actual_result"] == code
        reports.append(reliability_table(y_true, backtest_df[col], label))
    return pd.concat(reports, ignore_index=True)


def print_calibration_warning_if_needed(report: pd.DataFrame, threshold: float = 0.07):
    bad = report[report["gap"].abs() > threshold]
    if len(bad) == 0:
        print("Calibration check passed: no bins exceed the gap threshold.")
        return
    print(f"WARNING: {len(bad)} bin(s) show |predicted - observed| > {threshold:.0%}. "
          f"Consider Platt scaling / isotonic regression before simulating:")
    print(bad[["outcome", "bin", "n_matches", "predicted_mean", "observed_rate", "gap"]]
          .to_string(index=False))
