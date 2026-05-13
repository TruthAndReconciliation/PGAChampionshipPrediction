"""Training + scoring.

We blend two signals:

1. **GBR** regression on log(finish). Captures non-linear feature interactions
   when training data is rich (live ESPN data has full fields, finishes 1..70+).

2. **Heuristic** transparent linear score. Robust when training data is sparse
   (e.g. the bundled top-10-only CSV, which gives the GBR a censored target and
   pushes experienced players into the tail). Acts as a regularizer.

Blend weight `alpha` is set so the heuristic dominates when CV-MAE is poor.
Win probability = softmax over the blended score across the actual field.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR
from .features import FEATURE_COLS

log = logging.getLogger(__name__)

# Heuristic weights — higher = better expected finish. Hand-set, not learned.
# Lower returned value = better player. Course-fit terms are zero-effect when
# unavailable because the unknown-value sentinel (35) sits near the field median.
_H_WEIGHTS = {
    "recency_weighted_finish": +1.0,
    "top10_rate":              -8.0,
    "career_pga_champ_wins":   -2.0,
    "years_since_top10":       +0.25,
    "log_rank":                +0.6,
    "course_fit_score":        +0.35,  # blended scalar; lower = better fit
    "venue_avg_finish":        +0.10,  # exact venue history matters when present
}


def heuristic_score(X: pd.DataFrame) -> np.ndarray:
    """Return an array of pseudo-finish positions (lower = better)."""
    s = np.zeros(len(X))
    for col, w in _H_WEIGHTS.items():
        s += w * X[col].to_numpy()
    # Shift so the field median sits near a plausible finish (~30 in a major field).
    s = s - np.median(s) + 30.0
    return np.clip(s, 1.0, None)


@dataclass
class TrainedModel:
    pipeline: Pipeline
    cv_mae: float
    n_train: int
    blend_alpha: float  # weight on GBR; (1-alpha) on heuristic

    def save(self, path=MODEL_DIR / "pga_champ_model.joblib") -> None:
        joblib.dump({
            "pipeline": self.pipeline,
            "cv_mae": self.cv_mae,
            "n_train": self.n_train,
            "blend_alpha": self.blend_alpha,
        }, path)

    @classmethod
    def load(cls, path=MODEL_DIR / "pga_champ_model.joblib") -> "TrainedModel":
        blob = joblib.load(path)
        return cls(**blob)


def train(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> TrainedModel:
    """Fit GBR on log(finish). Group-CV by year prevents leakage."""
    y_log = np.log1p(y.to_numpy())
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("gbr",   GradientBoostingRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            random_state=42,
        )),
    ])

    cv_mae = float("nan")
    if groups is not None and groups.nunique() >= 3:
        n_splits = min(5, groups.nunique())
        gkf = GroupKFold(n_splits=n_splits)
        errs = []
        for tr, te in gkf.split(X, y_log, groups=groups):
            pipe.fit(X.iloc[tr], y_log[tr])
            pred = np.expm1(pipe.predict(X.iloc[te]))
            errs.append(np.mean(np.abs(pred - y.iloc[te].to_numpy())))
        cv_mae = float(np.mean(errs))
        log.info("Group-CV MAE (finish position) = %.2f", cv_mae)

    pipe.fit(X, y_log)

    # When the training target is censored (top-10 only), the heuristic is more
    # trustworthy. Detect by training-set max finish and adjust the blend.
    target_is_censored = y.max() <= 12
    blend_alpha = 0.25 if target_is_censored else 0.75
    if target_is_censored:
        log.info("Training target is censored (max finish=%s); leaning on heuristic (alpha=%.2f)",
                 int(y.max()), blend_alpha)

    return TrainedModel(pipeline=pipe, cv_mae=cv_mae, n_train=len(X), blend_alpha=blend_alpha)


def score_field(model: TrainedModel, X_pred: pd.DataFrame) -> pd.DataFrame:
    """Blend GBR + heuristic. Return ranked DataFrame with win probabilities."""
    pred_log = model.pipeline.predict(X_pred[FEATURE_COLS])
    gbr_finish = np.clip(np.expm1(pred_log), 1.0, None)
    heur_finish = heuristic_score(X_pred[FEATURE_COLS])

    a = model.blend_alpha
    blended = a * gbr_finish + (1 - a) * heur_finish

    # Softmax over -log(blended). Temperature controls how peaked the favorite is;
    # 0.6 keeps the chalk in a realistic ~8–15% band for a deep field.
    temperature = 0.6
    logits = -np.log1p(blended) / temperature
    weights = np.exp(logits - logits.max())
    win_prob = weights / weights.sum()

    out = pd.DataFrame({
        "player":           X_pred.index,
        "predicted_finish": blended,
        "win_probability":  win_prob,
    }).sort_values("predicted_finish").reset_index(drop=True)
    out["rank"] = out.index + 1
    return out[["rank", "player", "predicted_finish", "win_probability"]]
