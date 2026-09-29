import pandas as pd

# Sheet name in PL_Club_5years.xlsx -> canonical name (matches Opponent
# column values and Club_Stats_2627 "Squad" values)
CLUB_NAME_MAP = {
    "Arsenal": "Arsenal",
    "AstonVilla": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "CrystalPalace": "Crystal Palace",
    "Chelsea": "Chelsea",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich Town",
    "Leeds": "Leeds United",
    "Liverpool": "Liverpool",
    "ManCity": "Manchester City",
    "ManUnited": "Manchester Utd",
    "Newcastle": "Newcastle",
    "Nottingham": "Nottingham",
    "Spurs": "Tottenham",
    "Sunderland": "Sunderland",
}

# The 20 clubs in the 2026-27 season (from Club_Stats_2627). Coventry City
# and Hull City have NO historical data in PL_Club_5years.xlsx (freshly
# promoted, per the earlier transfer-network project's handling of the
# same two clubs) -- flagged explicitly wherever it matters rather than
# silently defaulted.
CLUBS_2026_27 = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Coventry City", "Crystal Palace", "Everton", "Fulham",
    "Hull City", "Ipswich Town", "Leeds United", "Liverpool",
    "Manchester City", "Manchester Utd", "Newcastle", "Nottingham",
    "Sunderland", "Tottenham",
]

CLUBS_WITHOUT_HISTORY = ["Coventry City", "Hull City"]


def _infer_season(date: pd.Timestamp) -> str:
    """PL seasons run Aug-May. A match in Jan-Jun YYYY belongs to the
    season that started the previous August."""
    year = date.year
    if date.month >= 7:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def load_historical_matches(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)

    # Load every sheet once, tag with canonical club name
    all_rows = {}
    for sheet in xls.sheet_names:
        if sheet not in CLUB_NAME_MAP:
            continue
        df = xls.parse(sheet)
        df["club"] = CLUB_NAME_MAP[sheet]
        all_rows[CLUB_NAME_MAP[sheet]] = df

    combined = pd.concat(all_rows.values(), ignore_index=True)
    combined["season"] = combined["Date"].apply(_infer_season)

    home_rows = combined[combined["Venue"] == "Home"].copy()
    away_rows = combined[combined["Venue"] == "Away"].copy()

    # Index away rows by (date, club, opponent) for lookup
    away_rows = away_rows.set_index(["Date", "club", "Opponent"])

    records = []
    missing_away = 0
    for row in home_rows.itertuples(index=False):
        key = (row.Date, row.Opponent, row.club)  # away side's (date, club=opponent, opponent=home club)
        result_map = {"Win": "H", "Draw": "D", "Lose": "A"}
        away_stats = None
        if key in away_rows.index:
            away_stats = away_rows.loc[key]
            if isinstance(away_stats, pd.DataFrame):  # duplicate safety
                away_stats = away_stats.iloc[0]

        records.append({
            "date": row.Date,
            "season": row.season,
            "home_team": row.club,
            "away_team": row.Opponent,
            "home_goals": row.GF,
            "away_goals": row.GA,
            "result": result_map.get(row.Result),
            "home_possession": row.Possession,
            "away_possession": away_stats["Possession"] if away_stats is not None else None,
            "home_shots": row.Shots,
            "away_shots": away_stats["Shots"] if away_stats is not None else None,
            "home_sot": row.SoT,
            "away_sot": away_stats["SoT"] if away_stats is not None else None,
        })
        if away_stats is None:
            missing_away += 1

    matches = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    if missing_away > 0:
        print(f"Note: {missing_away}/{len(matches)} matches couldn't be matched to the "
              f"away side's own sheet (likely a match against a club not in the 18 "
              f"available sheets, e.g. Leicester/Watford/Norwich/Southampton/Sheffield "
              f"Utd/Luton). Home-side data is complete for these; away-side stat "
              f"columns are NaN and excluded from feature engineering by fillna.")

    return matches


def load_current_season_table(club_stats_path: str, player_stats_path: str) -> pd.DataFrame:
    club_stats = pd.read_excel(club_stats_path, sheet_name="Club_Stats_2627")
    club_stats["points"] = club_stats["W"] * 3 + club_stats["D"]

    # Player_Stats_2627 has a 2-row header (see PL_26_27_Stats.xlsx quirk):
    # row 0 = ['Player','Pos','Squad', NaN...], row 1 = actual sub-headers
    # for the stat columns. Skip both and assign names manually.
    player_raw = pd.read_excel(player_stats_path, sheet_name="Player_Stats_2627",
                                header=None, skiprows=2)
    col_names = ["Player", "Pos", "Squad", "MP", "Starts", "Min", "90s",
                 "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
                 "Gls90", "Ast90", "GA90", "GPK90", "GAPK90"]
    player_raw.columns = col_names[:len(player_raw.columns)]
    player_raw["Gls"] = pd.to_numeric(player_raw["Gls"], errors="coerce").fillna(0)

    goals_for = player_raw.groupby("Squad")["Gls"].sum().rename("goals_for")

    table = club_stats.set_index("Squad")[["points", "MP", "GA"]].join(goals_for)
    table = table.rename(columns={"MP": "played", "GA": "goals_against"})
    table["goals_for"] = table["goals_for"].fillna(0)
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table = table.reset_index().rename(columns={"Squad": "team", "index": "team"})

    return table[["team", "points", "played", "goals_for", "goals_against", "goal_difference"]]
