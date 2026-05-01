import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# PART 1: HISTORICAL DATA WEIGHT OPTIMIZATION
# ============================================================

def optimize_weights(history_file):
    """
    Uses historical PGA Championship data to learn better feature weights.

    CSV must include:
    year, player, SG_total, SG_approach, recent_form, finish_position
    """

    history = pd.read_csv(history_file)

    history["top_10"] = history["finish_position"] <= 10

    features = ["SG_total", "SG_approach", "recent_form"]

    X = history[features]
    y = history["top_10"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression())
    ])

    model.fit(X, y)

    coefficients = model.named_steps["log_reg"].coef_[0]

    weights = pd.DataFrame({
        "feature": features,
        "raw_weight": coefficients,
        "absolute_importance": abs(coefficients)
    })

    weights["optimized_weight"] = (
        weights["absolute_importance"] /
        weights["absolute_importance"].sum()
    )

    return weights


# ============================================================
# PART 2: BUILD PLAYER RATINGS
# ============================================================

def build_player_ratings(players_df, weights_df):
    """
    Creates a player rating using optimized weights.
    """

    players_df = players_df.copy()

    weight_dict = dict(zip(
        weights_df["feature"],
        weights_df["optimized_weight"]
    ))

    players_df["rating"] = (
        weight_dict["SG_total"] * players_df["SG_total"] +
        weight_dict["SG_approach"] * players_df["SG_approach"] +
        weight_dict["recent_form"] * players_df["recent_form"]
    )

    return players_df


# ============================================================
# PART 3: CONVERT RATING TO EXPECTED SCORE
# ============================================================

def add_expected_scores(players_df, field_average=-8, scale=4):
    """
    Converts rating into a projected 72-hole score.

    Lower score = better.
    """

    players_df = players_df.copy()

    players_df["expected_score"] = (
        field_average - players_df["rating"] * scale
    )

    return players_df


# ============================================================
# PART 4: MONTE CARLO SIMULATION
# ============================================================

def simulate_tournament(players_df, sims=20000, random_seed=42):
    """
    Simulates the tournament many times.

    Needed columns:
    player, expected_score, volatility
    """

    np.random.seed(random_seed)

    win_counts = {player: 0 for player in players_df["player"]}
    top_10_counts = {player: 0 for player in players_df["player"]}

    for _ in range(sims):
        simulated_scores = np.random.normal(
            players_df["expected_score"],
            players_df["volatility"]
        )

        temp = players_df.copy()
        temp["simulated_score"] = simulated_scores
        temp = temp.sort_values("simulated_score")

        winner = temp.iloc[0]["player"]
        win_counts[winner] += 1

        top_10_players = temp.head(10)["player"]

        for player in top_10_players:
            top_10_counts[player] += 1

    results = pd.DataFrame({
        "player": players_df["player"],
        "expected_score": players_df["expected_score"],
        "win_pct": [
            win_counts[player] / sims * 100
            for player in players_df["player"]
        ],
        "top_10_pct": [
            top_10_counts[player] / sims * 100
            for player in players_df["player"]
        ]
    })

    return results.sort_values("win_pct", ascending=False)


# ============================================================
# PART 5: SAMPLE CURRENT PLAYER DATA
# ============================================================

players = pd.DataFrame({
    "player": [
        "Scottie Scheffler",
        "Rory McIlroy",
        "Jon Rahm",
        "Xander Schauffele",
        "Collin Morikawa",
        "Bryson DeChambeau",
        "Viktor Hovland",
        "Ludvig Aberg",
        "Patrick Cantlay",
        "Brooks Koepka"
    ],
    "SG_total": [
        2.5, 2.1, 2.0, 1.9, 1.8,
        1.7, 1.6, 1.5, 1.4, 1.3
    ],
    "SG_approach": [
        1.4, 1.2, 1.1, 1.0, 1.3,
        0.9, 1.0, 0.8, 0.9, 0.7
    ],
    "recent_form": [
        2.2, 1.8, 1.7, 1.6, 1.5,
        1.6, 1.2, 1.4, 1.1, 1.3
    ],
    "volatility": [
        2.8, 3.4, 3.5, 3.2, 3.0,
        4.0, 3.7, 3.9, 3.2, 4.1
    ]
})


# ============================================================
# PART 6: RUN EVERYTHING
# ============================================================

if __name__ == "__main__":

    # Option A:
    # Use optimized weights from historical data.
    #
    # Uncomment this when you have the CSV file:
    #
    # weights = optimize_weights("pga_championship_history.csv")

    # Option B:
    # Temporary fallback weights until you have historical data.

    weights = pd.DataFrame({
        "feature": ["SG_total", "SG_approach", "recent_form"],
        "optimized_weight": [0.60, 0.30, 0.10]
    })

    players = build_player_ratings(players, weights)

    players = add_expected_scores(
        players,
        field_average=-8,
        scale=4
    )

    results = simulate_tournament(
        players,
        sims=50000,
        random_seed=42
    )

    print("\nOptimized Weights:")
    print(weights)

    print("\nPlayer Ratings:")
    print(players[[
        "player",
        "rating",
        "expected_score",
        "volatility"
    ]].sort_values("rating", ascending=False))

    print("\nSimulation Results:")
    print(results)
