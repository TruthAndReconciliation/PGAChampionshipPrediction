"""Free, no-auth golf data fetchers.

Primary source: ESPN's public (undocumented but stable) JSON endpoints.
All responses are cached to disk so repeated runs are cheap and offline-friendly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Iterable

import requests

from .config import CACHE_DIR, ENDPOINTS, REQUEST_TIMEOUT, USER_AGENT

log = logging.getLogger(__name__)


def _cache_path(url: str, params: dict | None):
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def http_get(url: str, params: dict | None = None, force: bool = False) -> dict:
    """GET with on-disk cache. Returns parsed JSON or raises."""
    cp = _cache_path(url, params)
    if cp.exists() and not force:
        return json.loads(cp.read_text())

    log.info("GET %s params=%s", url, params)
    r = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    r.raise_for_status()
    payload = r.json()
    cp.write_text(json.dumps(payload))
    return payload


def get_season_events(year: int) -> list[dict]:
    """List every PGA tour event in a season. Used to locate PGA Championship IDs."""
    url = ENDPOINTS["season_events"].format(year=year)
    out: list[dict] = []
    page = 1
    while True:
        data = http_get(url, {"limit": 100, "page": page})
        for ref in data.get("items", []):
            # core API returns $ref objects; resolve each.
            if "$ref" in ref:
                out.append(http_get(ref["$ref"]))
            else:
                out.append(ref)
        if page >= data.get("pageCount", 1):
            break
        page += 1
        time.sleep(0.2)
    return out


def find_pga_championship(year: int) -> dict | None:
    """Return the ESPN event object for the PGA Championship in `year`, or None."""
    for ev in get_season_events(year):
        name = (ev.get("name") or "").lower()
        # ESPN labels it "PGA Championship"; guard against "Senior PGA" / "Women's PGA".
        if "pga championship" in name and "senior" not in name and "women" not in name:
            return ev
    return None


def get_leaderboard(event_id: str | int) -> dict:
    """Return the leaderboard payload for an event id (final round = final standings).

    The public ESPN doc uses ?tournamentId=<id>; older clients use ?event=<id>.
    Both currently work, but we prefer the documented form and fall back.
    """
    try:
        return http_get(ENDPOINTS["leaderboard"], {"tournamentId": str(event_id)})
    except Exception:
        return http_get(ENDPOINTS["leaderboard"], {"event": str(event_id)})


def get_venue(venue_id: str | int) -> dict:
    """Return venue (course) metadata. Used for course-fit features."""
    return http_get(ENDPOINTS["venue"].format(venue_id=venue_id))


def get_event(event_id: str | int) -> dict:
    """Return full event payload (includes venue ref and competition refs)."""
    return http_get(ENDPOINTS["event"].format(event_id=event_id))


def event_venue_id(event: dict) -> str | None:
    """Pull the venue id out of an ESPN event payload (handles $ref shapes)."""
    v = event.get("venue") or {}
    if isinstance(v, dict):
        if "id" in v:
            return str(v["id"])
        ref = v.get("$ref")
        if ref:
            # .../venues/<id>?... — last path segment before query string.
            return ref.rstrip("/").split("/")[-1].split("?")[0]
    return None


def get_current_rankings() -> list[dict]:
    """OWGR-equivalent rankings snapshot via ESPN."""
    data = http_get(ENDPOINTS["rankings"])
    # ESPN nests rankings in different shapes across releases; normalize defensively.
    ranks = data.get("rankings") or data.get("items") or []
    if ranks and isinstance(ranks[0], dict) and "ranks" in ranks[0]:
        ranks = ranks[0]["ranks"]
    return ranks


def iter_field(leaderboard: dict) -> Iterable[dict]:
    """Yield per-player rows from an ESPN leaderboard payload."""
    events = leaderboard.get("events") or []
    if not events:
        return
    competitions = events[0].get("competitions") or []
    if not competitions:
        return
    for comp in competitions[0].get("competitors", []):
        yield comp


def player_row(competitor: dict) -> dict:
    """Flatten an ESPN competitor blob to a plain dict."""
    ath = competitor.get("athlete", {}) or {}
    status = competitor.get("status", {}) or {}
    stats = {s["name"]: s.get("value") for s in competitor.get("statistics", []) or []}
    return {
        "athlete_id":   ath.get("id"),
        "name":         ath.get("displayName") or ath.get("fullName"),
        "country":      (ath.get("flag") or {}).get("alt") or ath.get("birthPlace", {}).get("country"),
        "position":     status.get("position", {}).get("id") or competitor.get("status", {}).get("position", {}).get("displayName"),
        "finish":       _parse_finish(competitor),
        "score_to_par": stats.get("scoreToPar") or competitor.get("score"),
        "total_strokes": stats.get("totalStrokes"),
        "made_cut":     not status.get("displayValue", "").upper().startswith("CUT"),
    }


def _parse_finish(competitor: dict) -> int | None:
    """Best-effort numeric finish position. Returns None for WD/MC/DQ."""
    s = competitor.get("status", {}) or {}
    pos = s.get("position", {}) or {}
    raw = pos.get("id") or pos.get("displayName") or ""
    raw = str(raw).lstrip("T").strip()
    if not raw.isdigit():
        return None
    return int(raw)


__all__ = [
    "http_get",
    "get_season_events",
    "find_pga_championship",
    "get_leaderboard",
    "get_current_rankings",
    "get_venue",
    "get_event",
    "event_venue_id",
    "iter_field",
    "player_row",
]
