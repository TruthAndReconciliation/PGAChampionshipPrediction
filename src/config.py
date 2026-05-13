"""Endpoint URLs and constants. All sources used here are free / no-auth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"

for d in (DATA_DIR, CACHE_DIR, RAW_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ESPN public (undocumented but widely used) endpoints. No key required.
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga"

ENDPOINTS = {
    "scoreboard":      f"{ESPN_SITE}/scoreboard",           # ?dates=YYYYMMDD
    "rankings":        f"{ESPN_SITE}/rankings",             # current OWGR snapshot
    "leaderboard":     f"{ESPN_SITE}/leaderboard",          # ?tournamentId=<id> (also accepts ?event=)
    "season_events":   f"{ESPN_CORE}/seasons/{{year}}/events",
    "event":           f"{ESPN_CORE}/events/{{event_id}}",
    "athlete":         f"{ESPN_CORE}/athletes/{{athlete_id}}",
    "venues":          f"{ESPN_CORE}/venues",               # course catalog
    "venue":           f"{ESPN_CORE}/venues/{{venue_id}}",
}

# PGA Championship has been the year's second major since 2019 (May).
# Training window: skip 2020 (held Aug, no fans) for venue/season comparability,
# but keep it as an optional toggle.
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
TARGET_YEAR = 2026

REQUEST_TIMEOUT = 15
USER_AGENT = "PGAChampionshipPrediction/0.1 (+github research)"
