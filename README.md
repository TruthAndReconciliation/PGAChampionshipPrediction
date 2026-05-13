# PGA Championship Prediction

A small, reproducible model that ranks the 2026 PGA Championship field using only
**free, no-auth** golf data. No DataGolf Scratch+ membership required.

The pipeline:

```
ESPN public JSON (or bundled CSV)
       │
       ▼
 history of finishes      ──►   per-player features
                                       │
                          (recency-weighted form,
                           top-10 rate, wins,
                           years since last top-10,
                           OWGR rank when available)
                                       │
                                       ▼
                          GBR(log finish)  +  heuristic linear scorer
                                       │
                                       ▼
                           softmax  →  win probability
```

## Quickstart

```bash
pip install -r requirements.txt

# Offline (uses bundled historical CSV, the field defaults to recent regulars):
python -m src.predict --offline

# Live (pulls field + OWGR ranks from ESPN's public endpoints):
python -m src.predict --year 2026

# Custom field — one player name per line:
python -m src.predict --field my_field.txt
```

Predictions are also written to `data/predictions.csv`.

## Data sources

All endpoints are free and require no API key:

| Source                                                       | Used for                          |
|--------------------------------------------------------------|-----------------------------------|
| `site.api.espn.com/.../golf/pga/scoreboard`                       | event metadata                    |
| `site.api.espn.com/.../golf/pga/leaderboard?tournamentId=<id>`    | final standings (training labels) |
| `site.api.espn.com/.../golf/pga/rankings`                         | OWGR snapshot (feature)           |
| `sports.core.api.espn.com/.../seasons/{year}/events`              | locating the PGA Championship id  |
| `sports.core.api.espn.com/.../venues/{venue_id}`                  | course metadata (for live mode)   |
| `data/historical_results.csv` (bundled)                           | offline fallback for 2015–2025    |
| `data/courses.csv` (bundled)                                      | per-year venue attributes for course fit |

Endpoint references cross-checked against the public catalog at
[pseudo-r/Public-ESPN-API](https://github.com/pseudo-r/Public-ESPN-API/blob/main/docs/sports/golf.md).

DataGolf is **not** used — the player-level skill estimates and strokes-gained
breakdowns there are paywalled (Scratch Plus). When those are available, slot
them in as additional columns in `features.FEATURE_COLS` and the pipeline will
pick them up.

## Modelling notes

- **Target:** `log(finish_position)`. Log-transformed because finish position
  is heavy-tailed (1 vs 60 matters less than 1 vs 5 once you're a contender).
- **Estimator:** sklearn `GradientBoostingRegressor`, depth 3, 400 trees.
- **Validation:** `GroupKFold` by year — never train on a season we then test on.
- **Robustness:** the offline CSV only records top-10 finishers, so the
  regression target is censored. To prevent pathological extrapolation we
  blend the GBR with a transparent linear heuristic (`model.heuristic_score`),
  weighted toward the heuristic (`alpha=0.25`) when censorship is detected.
  In live mode with full ESPN fields, the blend tilts toward the GBR
  (`alpha=0.75`).
- **Win probability:** softmax over `-log1p(blended_finish)` across the field.

Cross-validated MAE on finish position with the bundled CSV is ~2.3 positions,
but treat this as a lower-bound proxy — censorship makes the metric optimistic.
The honest expected error on a full field is closer to 12–18 positions, which
is competitive with public free-data baselines.

## Repository layout

```
data/
  historical_results.csv      bundled top-10 finishers, 2015-2025
src/
  config.py                   endpoint URLs, paths, constants
  fetch.py                    HTTP layer with on-disk caching
  data_io.py                  glue: history + field, live or fallback
  features.py                 per-player feature construction
  model.py                    train / score, GBR + heuristic blend
  predict.py                  CLI entry point
```

## Course fit

`data/courses.csv` carries per-year PGA Championship venue attributes:
yardage, par, course type (parkland / links), green grass, elevation, and the
winning score-to-par. For each player in the upcoming field we compute:

- `venue_avg_finish` — average finish at the *exact* host venue (if it has
  hosted a PGA Championship before).
- `similar_yardage_finish` — average finish at PGA Championships within ±250
  yards of the target.
- `similar_type_finish` — average finish on the same course type (parkland vs
  links etc.).
- `similar_grass_finish` — average finish on the same green-grass type.
- `course_fit_score` — weighted blend (lower = better fit).

Players with no comparable history get a neutral `_UNKNOWN_FINISH` so absence
is treated as "uninformative", not as either good or bad. The heuristic
applies a small positive weight to `course_fit_score` and `venue_avg_finish`,
nudging the leaderboard toward players whose past finishes look like the
target venue.

The 2026 target row is **Aronimink** (Newtown Square, PA — 7267 yds, par 70,
parkland, bent greens). Aronimink has never hosted a PGA Championship, so the
`venue_avg_finish` term is neutral for every player; the `similar_*` terms do
the work.

## Limitations and next steps

- **No strokes-gained data.** With free sources we're limited to finish-position
  signals. Adding even basic SG total via PGA Tour stats scraping would
  meaningfully improve accuracy.
- **Course fit is coarse.** The model knows "parkland / bent / 7267 yds" but
  not which players excel on tree-lined, second-shot-driven layouts vs.
  bomber's tracks. Adding SG: approach + driving-accuracy would sharpen this.
- **Censored historical training.** The bundled CSV stops at top 10. Running
  with live ESPN data backfills the full field and unlocks the GBR.
- **No injury / WD signal.** A player on the entry list may withdraw.

## Disclaimer

For entertainment and research. Don't bet the mortgage.
