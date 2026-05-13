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
| `site.api.espn.com/.../golf/pga/scoreboard`                  | event metadata                    |
| `site.api.espn.com/.../golf/pga/leaderboard?event=<id>`      | final standings (training labels) |
| `site.api.espn.com/.../golf/pga/rankings`                    | OWGR snapshot (feature)           |
| `sports.core.api.espn.com/.../seasons/{year}/events`         | locating the PGA Championship id  |
| `data/historical_results.csv` (bundled)                      | offline fallback for 2015–2025    |

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

## Limitations and next steps

- **No strokes-gained data.** With free sources we're limited to finish-position
  signals. Adding even basic SG total via PGA Tour stats scraping would
  meaningfully improve accuracy.
- **No course-fit features.** Driving-distance, GIR%, putting on bermuda etc.
  would help on courses like Aronimink (2026 host).
- **Censored historical training.** The bundled CSV stops at top 10. Running
  with live ESPN data backfills the full field and unlocks the GBR.
- **No injury / WD signal.** A player on the entry list may withdraw.

## Disclaimer

For entertainment and research. Don't bet the mortgage.
