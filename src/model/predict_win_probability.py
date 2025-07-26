import os
from catboost import CatBoostClassifier
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = CatBoostClassifier()
model.load_model(os.path.join(BASE_DIR, "model", "catboost_model.cbm"))

df = pd.read_csv(os.path.join(BASE_DIR, "output", "feature_dataset_light.csv"))

features = [
    'ranking_diff', 'rank_points_diff',
    'age_diff', 'height_diff',
    'same_hand', 'hand_matchup',
    'h2h_winrate', 'recent_winrate_diff'
]

def calculate_h2h_winrate(player1_name, player2_name):
    h2h_matches = df[
        ((df["winner_name"] == player1_name) & (df["loser_name"] == player2_name)) |
        ((df["winner_name"] == player2_name) & (df["loser_name"] == player1_name))
    ]
    if h2h_matches.empty:
        return 0.5
    wins = (h2h_matches["winner_name"] == player1_name).sum()
    return wins / len(h2h_matches)

def get_player_stats(player_name):
    records = df[(df["winner_name"] == player_name) | (df["loser_name"] == player_name)]
    if records.empty:
        raise ValueError(f"No records found for {player_name}")
    record = records.sort_values("tourney_date").iloc[-1]
    if record["winner_name"] == player_name:
        return {
            "rank": record["winner_rank"],
            "rank_points": record["winner_rank_points"],
            "age": record["winner_age"],
            "height": record["winner_ht"],
            "hand": record["winner_hand"],
            "recent_winrate": record["winner_recent_winrate"]
        }
    else:
        return {
            "rank": record["loser_rank"],
            "rank_points": record["loser_rank_points"],
            "age": record["loser_age"],
            "height": record["loser_ht"],
            "hand": record["loser_hand"],
            "recent_winrate": record["loser_recent_winrate"]
        }

def predict_win_probability(player1_name, player2_name):
    """
    双向预测稳定版
    """
    p1 = get_player_stats(player1_name)
    p2 = get_player_stats(player2_name)

    def build_features(a, b):
        return {
            'ranking_diff': b["rank"] - a["rank"],       # 保持方向性
            'rank_points_diff': a["rank_points"] - b["rank_points"],
            'age_diff': a["age"] - b["age"],
            'height_diff': a["height"] - b["height"],
            'same_hand': 1 if a["hand"] == b["hand"] else 0,
            'hand_matchup': f"{a['hand']}_{b['hand']}",
            'h2h_winrate': calculate_h2h_winrate(a_name, b_name),
            'recent_winrate_diff': a["recent_winrate"] - b["recent_winrate"]
        }

    # 第一次：p1 vs p2
    a_name, b_name = player1_name, player2_name
    X1 = pd.DataFrame([build_features(p1, p2)])[features]
    p1_prob_1 = model.predict_proba(X1)[0][1]

    # 第二次：p2 vs p1
    a_name, b_name = player2_name, player1_name
    X2 = pd.DataFrame([build_features(p2, p1)])[features]
    p2_prob_2 = model.predict_proba(X2)[0][1]

    # 平均化
    p1_prob = (p1_prob_1 + (1 - p2_prob_2)) / 2
    p2_prob = 1 - p1_prob

    return round(p1_prob, 4), round(p2_prob, 4)




if __name__ == "__main__":
    p1_prob, p2_prob = predict_win_probability("Novak Djokovic", "Jannik Sinner")
    print(f"Djokovic: {p1_prob:.2%}, Sinner: {p2_prob:.2%}")
