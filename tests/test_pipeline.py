"""Smoke tests — verify the offline pipeline runs end-to-end and produces sane output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_history_from_csv
from src.features import build_prediction_frame, build_training_frame, FEATURE_COLS
from src.model import score_field, train


def test_history_load():
    h = load_history_from_csv()
    assert len(h) > 50
    assert {"year", "finish", "name"}.issubset(h.columns)
    assert h["finish"].min() == 1


def test_training_frame_shape():
    h = load_history_from_csv()
    X, y, meta = build_training_frame(h, min_year=h["year"].min() + 2)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y) == len(meta)
    assert (y >= 1).all()


def test_end_to_end_offline():
    h = load_history_from_csv()
    X, y, meta = build_training_frame(h, min_year=h["year"].min() + 2)
    m = train(X, y, groups=meta["year"])

    field = ["Scottie Scheffler", "Brooks Koepka", "Rory McIlroy",
             "Xander Schauffele", "Bryson DeChambeau"]
    Xp = build_prediction_frame(field, h, asof_year=2026)
    board = score_field(m, Xp)

    assert len(board) == len(field)
    assert set(board["player"]) == set(field)
    # Probabilities sum to 1 over the field.
    assert abs(board["win_probability"].sum() - 1.0) < 1e-6
    # All predicted finishes are >= 1.
    assert (board["predicted_finish"] >= 1.0).all()
    # Players with real PGA Champ records (Koepka, Scheffler) should outrank
    # someone with zero starts in this dataset (Rory).
    rank_of = {row.player: row.rank for row in board.itertuples()}
    assert rank_of["Brooks Koepka"] < rank_of["Rory McIlroy"]
    assert rank_of["Scottie Scheffler"] < rank_of["Rory McIlroy"]


if __name__ == "__main__":
    test_history_load()
    test_training_frame_shape()
    test_end_to_end_offline()
    print("ok")
