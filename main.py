from dataclasses import dataclass
import numpy as np
import pandas as pd


# ============================================================
# PLAYER STRUCTURE
# ============================================================

@dataclass
class Player:
    name: str
    sg_total: float
    sg_approach: float
    recent_form: float
    volatility: float
    rating: float = 0.0
    expected_score: float = 0.0

    def calculate_rating(self, weights):
        self.rating = (
            weights["SG_total"] * self.sg_total +
            weights["SG_approach"] * self.sg_approach +
            weights["recent_form"] * self.recent_form
        )

    def calculate_expected_score(self, field_average=-8, scale=4):
        self.expected_score = field_average - self.rating * scale


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def players_to_dataframe(players):
    return pd.DataFrame([
        {
            "player": p.name,
            "SG_total": p.sg_total,
            "SG_approach": p.sg_approach,
            "recent_form": p.recent_form,
            "volatility": p.volatility,
            "rating": p.rating,
            "expected_score": p.expected_score
        }
        for p in players
    ])


def simulate_tournament(players_df, sims=50000, random_seed=42):
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
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    weights = {
        "SG_total": 0.60,
        "SG_approach": 0.30,
        "recent_form": 0.10
    }

    players = [
        Player("Scottie Scheffler", 2.5, 1.4, 2.2, 2.8),
        Player("Rory McIlroy", 2.1, 1.2, 1.8, 3.4),
        Player("Jon Rahm", 2.0, 1.1, 1.7, 3.5),
        Player("Xander Schauffele", 1.9, 1.0, 1.6, 3.2),
        Player("Collin Morikawa", 1.8, 1.3, 1.5, 3.0),
        Player("Bryson DeChambeau", 1.7, 0.9, 1.6, 4.0),
        Player("Viktor Hovland", 1.6, 1.0, 1.2, 3.7),
        Player("Ludvig Aberg", 1.5, 0.8, 1.4, 3.9),
        Player("Patrick Cantlay", 1.4, 0.9, 1.1, 3.2),
        Player("Brooks Koepka", 1.3, 0.7, 1.3, 4.1),
    ]

    for player in players:
        player.calculate_rating(weights)
        player.calculate_expected_score(field_average=-8, scale=4)

    players_df = players_to_dataframe(players)

    results = simulate_tournament(players_df)

    print("\nPlayer Data:")
    print(players_df.sort_values("rating", ascending=False))

    print("\nSimulation Results:")
    print(results)
