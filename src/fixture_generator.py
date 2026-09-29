import pandas as pd


def generate_round_robin(clubs: list[str]) -> list[list[tuple[str, str]]]:
    n = len(clubs)
    assert n % 2 == 0, "circle method requires an even number of clubs"

    teams = clubs.copy()
    fixed = teams[0]
    rotating = teams[1:]

    first_half_rounds = []
    for _ in range(n - 1):
        round_teams = [fixed] + rotating
        pairs = []
        for i in range(n // 2):
            home, away = round_teams[i], round_teams[n - 1 - i]
            pairs.append((home, away))
        first_half_rounds.append(pairs)
        rotating = [rotating[-1]] + rotating[:-1]

    # Alternate home/away within each round so one team doesn't get all
    # home games in the first leg -- standard practical tweak, doesn't
    # affect correctness of "each pair plays once per half."
    first_half_rounds = [
        [(h, a) if i % 2 == 0 else (a, h) for i, (h, a) in enumerate(rnd)]
        for i, rnd in enumerate(first_half_rounds)
    ]

    second_half_rounds = [[(a, h) for h, a in rnd] for rnd in first_half_rounds]

    return first_half_rounds + second_half_rounds


def generate_remaining_fixtures_df(clubs: list[str], already_played_rounds: int = 4) -> pd.DataFrame:
    rounds = generate_round_robin(clubs)
    remaining_rounds = rounds[already_played_rounds:]

    rows = []
    for i, rnd in enumerate(remaining_rounds, start=already_played_rounds + 1):
        for home, away in rnd:
            rows.append({"gameweek": i, "home_team": home, "away_team": away})
    return pd.DataFrame(rows)
