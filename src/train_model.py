import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

RESULT_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_RESULT = {v: k for k, v in RESULT_TO_INT.items()}

TEST_SEASON = "2025-26"  # most recent complete season held out for evaluation


def train_and_evaluate(train_df: pd.DataFrame, feature_cols: list[str]):
    train_mask = train_df["season"] != TEST_SEASON
    test_mask = train_df["season"] == TEST_SEASON

    X_train = train_df.loc[train_mask, feature_cols].values
    y_train = train_df.loc[train_mask, "result"].map(RESULT_TO_INT).values
    X_test = train_df.loc[test_mask, feature_cols].values
    y_test = train_df.loc[test_mask, "result"].map(RESULT_TO_INT).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)
    results["LogisticRegression"] = {
        "model": logreg,
        "needs_scaling": True,
        "accuracy": accuracy_score(y_test, logreg_proba.argmax(axis=1)),
        "logloss": log_loss(y_test, logreg_proba),
        "proba": logreg_proba,
    }

    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)
    results["XGBoost"] = {
        "model": xgb,
        "needs_scaling": False,
        "accuracy": accuracy_score(y_test, xgb_proba.argmax(axis=1)),
        "logloss": log_loss(y_test, xgb_proba),
        "proba": xgb_proba,
    }

    print("Model comparison (test season: {}):".format(TEST_SEASON))
    baseline_acc = max(np.bincount(y_test)) / len(y_test)
    print(f"  Baseline (always predict majority class): {baseline_acc:.3f}")
    for name, r in results.items():
        print(f"  {name}: accuracy={r['accuracy']:.3f}, log_loss={r['logloss']:.3f}")

    best_name = min(results, key=lambda n: results[n]["logloss"])
    best = results[best_name]
    print(f"\nSelected: {best_name} (lowest log_loss -- log_loss rewards "
          f"calibrated probabilities, which matters more here than raw "
          f"accuracy since the simulator consumes probabilities directly)")

    test_predictions = train_df.loc[test_mask, ["date", "home_team", "away_team", "result"]].copy()
    test_predictions["p_home_win"] = best["proba"][:, 0]
    test_predictions["p_draw"] = best["proba"][:, 1]
    test_predictions["p_away_win"] = best["proba"][:, 2]
    test_predictions = test_predictions.rename(columns={"result": "actual_result"})

    return best["model"], scaler, best_name, best["needs_scaling"], best["accuracy"], best["logloss"], test_predictions


def predict_fixture_probabilities(model, scaler, needs_scaling: bool,
                                   fixtures_with_features: pd.DataFrame,
                                   feature_cols: list[str]) -> pd.DataFrame:
    X = fixtures_with_features[feature_cols].values
    if needs_scaling:
        X = scaler.transform(X)
    proba = model.predict_proba(X)

    out = fixtures_with_features[["gameweek", "home_team", "away_team"]].copy()
    out["p_home_win"] = proba[:, 0]
    out["p_draw"] = proba[:, 1]
    out["p_away_win"] = proba[:, 2]
    return out
