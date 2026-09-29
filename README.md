# Premier League Season Simulator

A Monte Carlo season simulation engine built on top of match-outcome
probabilities from a trained model (XGBoost / Logistic Regression). Instead
of predicting a single result per match, it simulates the remainder of the
season thousands of times to produce probability distributions over final
league position: title odds, top-4 odds, relegation odds, and expected
points with confidence bands.

This is a companion project to the FPL match-prediction pipeline and the
Premier League transfer-network analysis — it reuses the same club-level
match data and probability outputs, extended into season-level simulation.

## Why simulate instead of just using the raw per-match probabilities?

A single match's predicted probability doesn't tell you much about season
outcomes on its own — errors don't cancel out linearly across 38 games,
remaining fixture difficulty is asymmetric between clubs, and small
per-match edges can swing title or relegation races nonlinearly. Monte
Carlo simulation captures all of this correctly and — critically — reports
*uncertainty*, not just a point estimate.

## Pipeline

1. **Calibration check** (`src/calibration.py`) — verifies the model's
   predicted probabilities match observed outcome frequencies on a
   held-out backtest set. Simulation is very sensitive to miscalibration:
   an overconfident model produces simulated tables with unrealistically
   narrow variance. Fix via `CalibratedClassifierCV` (Platt scaling or
   isotonic regression) on the underlying model before simulating if the
   report flags a gap.
2. **Simulation** (`src/simulate.py`) — for N runs, draws a W/D/L outcome
   per remaining fixture from the model's probability triple, accumulates
   points on top of the current real table, and records the final position
   of every club in every run.
3. **Aggregation** (`src/aggregate.py`) — turns raw simulation output into
   a club-level summary table (title/top4/top6/relegation probabilities,
   expected points with a 10th-90th percentile band) and a full
   team-by-position probability matrix.

## Scope: Option A vs Option B

This repo implements **Option A**: W/D/L-only simulation. Ties on points
are broken with a negligible random tiebreak jitter (or optionally by
current goal difference — see `SimulationConfig.tiebreak`), since goal
difference itself isn't simulated.

**Option B** (not yet implemented, planned extension) would add a
Dixon-Coles-style bivariate Poisson goals model so scorelines — and
therefore accurate goal-difference tiebreaks — are simulated directly,
rather than approximated. This would also unlock score-level outputs like
"most likely correct score" and "margin needed to survive." The current
module boundaries (`schema.py` / `simulate.py` / `aggregate.py`) are
structured so this can be added without rewriting the rest of the
pipeline — `simulate.py` is the only file that would need a
scoreline-aware variant.

## Two pipelines: synthetic demo vs. real data

- `src/main.py` — runs on synthetic data (`synthetic_data.py`), for
  quickly exercising the simulator without any real files.
- `src/main_real_data.py` — the real pipeline, using
  `data/PL_Club_5years.xlsx` and `data/PL_26_27_Stats.xlsx`. This is the
  one to use going forward.

### What the real pipeline does

1. **`data_loader.py`** — `PL_Club_5years.xlsx` stores one row per match
   *per club* (each match appears twice, once in each side's sheet).
   `load_historical_matches()` collapses this into one row per fixture,
   pulling each side's own reported stats (possession, shots, etc.) from
   that side's own sheet rather than guessing. `load_current_season_table()`
   builds the current 2026-27 table from `Club_Stats_2627` (which has
   W/D/L and goals *against* but not goals *for* — goals for is
   reconstructed by summing player-level goals from `Player_Stats_2627`).
2. **`features.py`** — rolling team-strength features (points, goals
   for/against, goal difference, possession, shots) via EWM recency
   weighting, matching the approach used in the companion FPL
   prediction pipeline: an 8-match span, reset (shrunk 50%, not
   wiped) at each season boundary, `.shift(1)`'d so no match ever sees
   its own result.
3. **`train_model.py`** — trains LogisticRegression and XGBoost on the
   engineered features, evaluated with a **season-based** train/test
   split (train on 2021-22 through 2024-25, test on 2025-26) rather than
   a random split, since random splitting would leak future form into
   training. Selects by log loss, not accuracy, because the simulator
   consumes probabilities directly — a well-calibrated model with
   slightly lower accuracy is more useful here than an overconfident one
   with slightly higher accuracy.
4. **`current_strength.py`** — builds each club's current-form feature
   vector as of today, blending shrunk end-of-2025-26 form with the 4
   games played so far this season.
5. **`fixture_generator.py`** — generates the remaining schedule.
6. Then: predict probabilities on the generated fixtures, calibration
   check, Monte Carlo simulate, aggregate — same as the synthetic
   pipeline from here.

### Two approximations worth knowing about (clearly flagged in code)

- **No real remaining fixture list was in the uploaded data** — only
  season-to-date aggregates. `fixture_generator.py` generates a
  plausible double round-robin (circle method) for the remaining 34
  gameweeks instead. Swap in the real fixture list (Premier League's
  official site or football-data.co.uk) as soon as it's available —
  nothing downstream needs to change.
- **No match-by-match detail for the 4 games played this season** —
  only season-to-date aggregates (W/D/L, goals against, and goals for
  via summed player stats). `current_strength.py` approximates this as
  4 identical sequential EWM updates using the season-to-date per-game
  average, rather than 4 distinct updates from the actual 4 results.
  Possession/shots aren't in the 2026-27 aggregate files at all, so
  those two features are carried forward unchanged from last season.
- **Coventry City and Hull City have zero historical data** in
  `PL_Club_5years.xlsx` (freshly promoted — same situation already
  handled in the companion transfer-network project). Their prior-season seed
  uses the average end-of-2025-26 form of last season's bottom-3 clubs
  as a "newly promoted, expect to be near the bottom" prior, then blends
  in their real 4-game 2026-27 stats on top.
- Also worth noting: some opponents in the historical match log (Wolves,
  Burnley, West Ham, and other clubs not among the current 20) don't
  have their own sheet in `PL_Club_5years.xlsx`, so matches against them
  are still included (home side's data is complete) but their own-side
  stats are NaN and excluded from feature engineering — this is handled
  automatically, not something you need to fix.

### First real-data run: what it showed

Model: LogisticRegression beat XGBoost on log loss (0.496 accuracy vs.
0.432 majority-class baseline on the 2025-26 held-out season) — in the
same ballpark as the FPL prediction pipeline's reported 0.404. The
calibration check flagged a handful of bins with gaps over 7 percentage
points — expected at this test-set size (some bins have under 5 matches)
but worth revisiting with more data or a calibration layer
(`CalibratedClassifierCV`) before treating title/relegation probabilities
as precise. Man City and Arsenal came out as the clear top-2 by title
probability, with Brighton a distant third given a strong 4-game start;
Coventry City was the clear relegation favorite.

## Validation approach (recommended before trusting outputs)

Backtest on a completed season: simulate from, say, gameweek 10 onward
using only pre-gameweek-10 data and your trained model, then check whether
the actual final table fell within the simulated distributions, and
whether events assigned ~X% probability across many such backtests
actually happened ~X% of the time. Compare against a naive baseline
(current points + games-remaining × historical points-per-game) to confirm
the model adds value over the obvious heuristic.

## Data attribution

Historical match data sourced from FBref.com, consistent with the
attribution approach used in the companion transfer-network and FPL
pipeline repos — raw scraped data is not redistributed in this repo, only
derived/aggregated outputs.

## Running

```bash
pip install -r requirements.txt
cd src
python main.py
```

Outputs land in `outputs/`: `calibration_report.csv`,
`season_probabilities.csv`, `position_distribution.csv`.

Run tests with:

```bash
python -m pytest tests/
```

## Project structure

```
pl-season-simulator/
├── data/
│   ├── PL_Club_5years.xlsx      # 5 seasons of match-by-match club data
│   ├── PL_Players_5Years.xlsx   # 5 seasons of player season stats
│   └── PL_26_27_Stats.xlsx      # current season-to-date aggregates
├── src/
│   ├── schema.py             # data contract between model output and simulator
│   ├── calibration.py        # probability calibration checks
│   ├── simulate.py           # Monte Carlo simulation engine
│   ├── aggregate.py          # raw sim output -> probability tables
│   ├── synthetic_data.py     # demo data generator (for main.py)
│   ├── main.py               # synthetic-data pipeline entry point
│   ├── data_loader.py        # loads + unifies the real xlsx data
│   ├── features.py           # rolling EWM team-form feature engineering
│   ├── train_model.py        # trains + evaluates the real outcome model
│   ├── current_strength.py   # current-season team strength (real data)
│   ├── fixture_generator.py  # round-robin remaining-fixture generator
│   └── main_real_data.py     # real-data pipeline entry point (use this one)
├── tests/
│   └── test_simulate.py
├── outputs/                   # generated CSVs land here
├── requirements.txt
└── README.md
```

## Planned extensions

- Dixon-Coles scoreline simulation (Option B — accurate GD tiebreaks,
  correct-score outputs)
- Weekly automated re-simulation as real results come in, tracking how
  title/relegation probabilities evolve over the season
- Interactive dashboard (Streamlit) for manually testing "what if X result
  happens" scenarios
- FPL crossover: combine with player-level predictions to flag players on
  teams with rising title/top-4 probability as differentials
