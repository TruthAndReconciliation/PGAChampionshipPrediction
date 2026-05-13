"""Feature engineering for PGA Championship outcome prediction.

Features are computed purely from per-event finish positions plus (when
available) a world-ranking snapshot. The design constraint: every feature
must be computable from free, no-auth sources.

Per-player features (as of an `asof_year`):
    * career_pga_champ_starts       count of prior PGA Championship starts
    * career_pga_champ_top10s       prior top-10s
    * career_pga_champ_wins         prior wins
    * top10_rate                    top10s / starts (smoothed)
    * recency_weighted_finish       exp-decayed avg finish in prior PGA Champs
    * best_finish_5yr               best finish in the last five years
    * years_since_top10             null if never
    * world_rank                    OWGR rank at the time (optional; 999 if unknown)
    * log_rank                      log1p(world_rank)
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "career_pga_champ_starts",
    "career_pga_champ_top10s",
    "career_pga_champ_wins",
    "top10_rate",
    "recency_weighted_finish",
    "best_finish_5yr",
    "years_since_top10",
    "world_rank",
    "log_rank",
]

_UNKNOWN_RANK = 999
_UNKNOWN_YEARS = 25  # cap for "years since top 10"


def _player_history_features(history: pd.DataFrame, asof_year: int) -> dict:
    """Compute per-player features from this player's prior finishes only."""
    prior = history[history["year"] < asof_year]
    starts = len(prior)
    top10s = int((prior["finish"] <= 10).sum()) if starts else 0
    wins = int((prior["finish"] == 1).sum()) if starts else 0

    # Beta-smoothed top-10 rate (alpha=1, beta=4 → prior mean ~0.20).
    top10_rate = (top10s + 1) / (starts + 5)

    if starts == 0:
        rec_weight, best5, years_since = 50.0, 50, _UNKNOWN_YEARS
    else:
        # exponential decay: most recent year weight 1.0, drop 30% per year.
        deltas = asof_year - prior["year"].to_numpy()
        w = np.exp(-0.3 * deltas)
        rec_weight = float(np.average(prior["finish"], weights=w))

        last5 = prior[prior["year"] >= asof_year - 5]
        best5 = int(last5["finish"].min()) if len(last5) else 50

        tops = prior[prior["finish"] <= 10]
        years_since = int(asof_year - tops["year"].max()) if len(tops) else _UNKNOWN_YEARS

    return {
        "career_pga_champ_starts": starts,
        "career_pga_champ_top10s": top10s,
        "career_pga_champ_wins":   wins,
        "top10_rate":              top10_rate,
        "recency_weighted_finish": rec_weight,
        "best_finish_5yr":         best5,
        "years_since_top10":       years_since,
    }


def _attach_rank(row: dict, ranks: dict | None, name: str) -> dict:
    r = (ranks or {}).get(_norm(name), _UNKNOWN_RANK)
    row["world_rank"] = r
    row["log_rank"] = math.log1p(r)
    return row


def _norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


def build_training_frame(
    history: pd.DataFrame,
    min_year: int = 2017,
    ranks_by_year: dict[int, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Per (year, player) row with features computed as-of that year.

    Returns:
        X: feature matrix
        y: target finish position (numeric, ties treated as the listed position)
        meta: year + player identifiers, aligned with X
    """
    rows, targets, meta = [], [], []
    years = sorted(history["year"].unique())
    for yr in years:
        if yr < min_year:
            continue
        ranks = (ranks_by_year or {}).get(yr)
        subset = history[history["year"] == yr]
        for _, r in subset.iterrows():
            player_hist = history[history["name"] == r["name"]]
            feats = _player_history_features(player_hist, asof_year=yr)
            feats = _attach_rank(feats, ranks, r["name"])
            rows.append(feats)
            targets.append(int(r["finish"]))
            meta.append({"year": yr, "name": r["name"]})

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(targets, name="finish")
    return X, y, pd.DataFrame(meta)


def build_prediction_frame(
    field: Iterable[str],
    history: pd.DataFrame,
    asof_year: int,
    ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Feature matrix for the upcoming field. Index is the player name."""
    rows = {}
    for name in field:
        player_hist = history[history["name"] == name]
        feats = _player_history_features(player_hist, asof_year=asof_year)
        feats = _attach_rank(feats, ranks, name)
        rows[name] = feats
    return pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_COLS)


def normalize_rank_map(rankings: list[dict]) -> dict[str, int]:
    """Build {normalized_name: rank} from ESPN rankings payload entries."""
    out: dict[str, int] = {}
    for entry in rankings:
        ath = entry.get("athlete") or {}
        name = ath.get("displayName") or ath.get("fullName") or entry.get("displayName")
        rank = entry.get("current") or entry.get("rank") or entry.get("currentRanking")
        if name and isinstance(rank, (int, float)):
            out[_norm(name)] = int(rank)
    return out
