import pandas as pd


def summarize(
    sim_results: pd.DataFrame,
    relegation_cutoff: int = 18,   # position >= this = relegated (20-team league)
    top4_cutoff: int = 4,
    top6_cutoff: int = 6,
) -> pd.DataFrame:
    n_sims = sim_results["sim_id"].nunique()

    points_summary = sim_results.groupby("team")["final_points"].agg(
        expected_points="mean",
        points_p10=lambda s: s.quantile(0.10),
        points_p90=lambda s: s.quantile(0.90),
    )

    def event_rate(mask_col, condition):
        counts = sim_results[condition].groupby("team").size()
        return (counts / n_sims).reindex(points_summary.index, fill_value=0.0)

    p_title = event_rate("final_position", sim_results["final_position"] == 1)
    p_top4 = event_rate("final_position", sim_results["final_position"] <= top4_cutoff)
    p_top6 = event_rate("final_position", sim_results["final_position"] <= top6_cutoff)
    p_relegation = event_rate("final_position", sim_results["final_position"] >= relegation_cutoff)

    summary = points_summary.copy()
    summary["p_title"] = p_title
    summary["p_top4"] = p_top4
    summary["p_top6"] = p_top6
    summary["p_relegation"] = p_relegation

    return summary.sort_values("expected_points", ascending=False).round(
        {"expected_points": 1, "points_p10": 0, "points_p90": 0,
         "p_title": 3, "p_top4": 3, "p_top6": 3, "p_relegation": 3}
    )


def position_distribution(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a team x position matrix of probabilities -- e.g. the
    probability each club finishes in each of the 20 positions.
    Useful for a heatmap visualization.
    """
    n_sims = sim_results["sim_id"].nunique()
    dist = (
        sim_results.groupby(["team", "final_position"])
        .size()
        .unstack(fill_value=0)
        / n_sims
    )
    return dist.round(3)
