# Raw data (not included)

The raw match and player statistics used in this project were sourced from
[FBref.com](https://fbref.com) (Sports Reference LLC). Per FBref's Terms of Use,
data sharing/reuse is welcome with clear attribution, but bulk redistribution of a
materially significant portion of their underlying data is restricted — so the raw
Excel exports are **not included in this repo**.

To reproduce this project:
1. Pull the equivalent match/player data yourself from FBref, respecting their
   [rate limits and terms of use](https://static.fbref.com/termsofuse.html)
   (e.g. via [`fbrefdata`](https://pypi.org/project/fbrefdata/) or a similarly
   rate-limited scraper).
2. Save it here as `PL_Club_5years.xlsx` and `PL_Players_5Years.xlsx`, matching the
   sheet structure expected by `src/01_build_distance_dataset.py` (one sheet per
   club, columns including `Date`, `Venue`, `Opponent`, `Result`, `GF`, `GA`,
   `Possession`, `Shots`, `SoT`, `Crosses`, `Interceptions`, `Offsides`,
   `Opp Shots`, `Formation`).

All derived outputs in `data/processed/` and `outputs/` are original analysis
(distance calculations, similarity scores, cluster assignments) built on top of
that data, not a republication of it, and are shared freely.

**Data source: [FBref.com](https://fbref.com) / Sports Reference LLC.**
