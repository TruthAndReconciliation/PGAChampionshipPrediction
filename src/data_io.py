"""Glue layer: load historical results + current field, from API or bundled CSV."""
from __future__ import annotations

import logging

import pandas as pd

from . import fetch
from .config import DATA_DIR, TRAIN_YEARS
from .features import normalize_rank_map

log = logging.getLogger(__name__)

HISTORICAL_CSV = DATA_DIR / "historical_results.csv"


def load_history_from_csv() -> pd.DataFrame:
    return pd.read_csv(HISTORICAL_CSV)


def load_history_from_espn(years=TRAIN_YEARS) -> pd.DataFrame:
    """Fetch top finishers per year from ESPN. Falls back to CSV on any failure."""
    rows = []
    for yr in years:
        try:
            ev = fetch.find_pga_championship(yr)
            if not ev:
                log.warning("No PGA Championship event found for %s", yr)
                continue
            lb = fetch.get_leaderboard(ev["id"])
            for comp in fetch.iter_field(lb):
                r = fetch.player_row(comp)
                if r["finish"] is None:
                    continue
                rows.append({
                    "year":   yr,
                    "finish": r["finish"],
                    "name":   r["name"],
                    "score_to_par": r["score_to_par"],
                })
        except Exception as e:  # network, parse, etc.
            log.warning("ESPN fetch failed for %s: %s", yr, e)
    if not rows:
        log.warning("ESPN history empty → falling back to bundled CSV")
        return load_history_from_csv()
    return pd.DataFrame(rows)


def load_history(prefer_live: bool = True) -> pd.DataFrame:
    if prefer_live:
        df = load_history_from_espn()
        if not df.empty:
            return df
    return load_history_from_csv()


def load_current_field(year: int) -> tuple[list[str], dict[str, int]]:
    """Return (field_player_names, normalized_rank_map). Empty list if unavailable."""
    try:
        ev = fetch.find_pga_championship(year)
        if not ev:
            return [], {}
        lb = fetch.get_leaderboard(ev["id"])
        field = [fetch.player_row(c)["name"] for c in fetch.iter_field(lb)]
        field = [n for n in field if n]
    except Exception as e:
        log.warning("ESPN field fetch failed: %s", e)
        field = []

    try:
        ranks = normalize_rank_map(fetch.get_current_rankings())
    except Exception as e:
        log.warning("ESPN rankings fetch failed: %s", e)
        ranks = {}
    return field, ranks
