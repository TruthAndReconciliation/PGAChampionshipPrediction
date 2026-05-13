"""End-to-end CLI: load data → train model → predict the target year's field.

Usage:
    python -m src.predict                       # use ESPN if reachable, else bundled CSV
    python -m src.predict --offline             # force the bundled CSV path
    python -m src.predict --year 2026           # explicit target
    python -m src.predict --field path/to.txt   # one player name per line
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, TARGET_YEAR
from .data_io import load_courses, load_current_field, load_history, load_history_from_csv
from .features import build_prediction_frame, build_training_frame
from .model import score_field, train


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _read_field_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _default_field(history: pd.DataFrame, top_n: int = 60) -> list[str]:
    """Best-effort field: take every player with at least one top-25 in last 5 yrs."""
    recent = history.sort_values("year", ascending=False)
    names: list[str] = []
    for n in recent["name"]:
        if n not in names:
            names.append(n)
        if len(names) >= top_n:
            break
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict the PGA Championship.")
    p.add_argument("--year", type=int, default=TARGET_YEAR)
    p.add_argument("--offline", action="store_true", help="Skip live ESPN fetches.")
    p.add_argument("--field", type=Path, help="File with one player name per line.")
    p.add_argument("--out", type=Path, default=DATA_DIR / "predictions.csv")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)
    log = logging.getLogger("predict")

    history = load_history_from_csv() if args.offline else load_history(prefer_live=True)
    courses = load_courses()
    log.info("History: %s rows across %s seasons", len(history), history["year"].nunique())
    log.info("Course metadata for %s seasons (incl. target)", len(courses))

    X, y, meta = build_training_frame(
        history, min_year=history["year"].min() + 2, courses=courses,
    )
    log.info("Training rows: %s", len(X))
    model = train(X, y, groups=meta["year"])
    log.info("Cross-val MAE on finish position: %.2f", model.cv_mae)

    if args.field:
        field = _read_field_file(args.field)
        ranks: dict[str, int] = {}
    elif args.offline:
        field, ranks = _default_field(history), {}
    else:
        field, ranks = load_current_field(args.year)
        if not field:
            log.warning("Live field unavailable; using historical regulars as a stand-in.")
            field = _default_field(history)

    X_pred = build_prediction_frame(
        field, history, asof_year=args.year, ranks=ranks, courses=courses,
    )
    board = score_field(model, X_pred)

    target_course = courses[courses["year"] == args.year]
    if len(target_course):
        tc = target_course.iloc[0]
        log.info("Target venue: %s (%s yds, par %s, %s, %s greens)",
                 tc["course"], tc["yardage"], tc["par"], tc["course_type"], tc["grass_green"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out, index=False)

    print(f"\nPredicted leaderboard — PGA Championship {args.year}")
    print(f"(model cv-MAE on finish position: {model.cv_mae:.2f})\n")
    show = board.head(20).copy()
    show["win_probability"] = (show["win_probability"] * 100).map(lambda v: f"{v:5.1f}%")
    show["predicted_finish"] = show["predicted_finish"].map(lambda v: f"{v:5.1f}")
    print(show.to_string(index=False))
    print(f"\nFull leaderboard written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
