import streamlit as st
import pandas as pd
from urllib.parse import quote_plus
from src.model.predict_win_probability import predict_win_probability, get_player_stats, calculate_h2h_winrate

st.set_page_config(layout="wide")
st.title("Player Comparison")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("src/output/feature_dataset_light.csv")
players_2024 = df[df['tourney_date'].astype(str).str.startswith('2024')]
unique_players = sorted(set(players_2024['winner_name']).union(set(players_2024['loser_name'])))


query_params = st.query_params
default_player = query_params.get("player", None)

index_p1 = 0
if default_player and default_player in unique_players:
    index_p1 = unique_players.index(default_player)

st.title("Player Comparison")

col1, col2 = st.columns(2)

with col1:
    player1 = st.selectbox("Select Player 1", unique_players, index=index_p1)

with col2:
    default_index_p2 = 1 if index_p1 == 0 else 0
    player2 = st.selectbox("Select Player 2", unique_players, index=default_index_p2)

# -------------------------------
# Player stats
# -------------------------------
col1, col2 = st.columns(2)

stats_p1 = get_player_stats(player1)
stats_p2 = get_player_stats(player2)

with col1:
    st.subheader(player1)
    st.markdown(f"""
    - **Rank:** {stats_p1["rank"]}
    - **Age:** {int(stats_p1["age"])}
    - **Height:** {stats_p1["height"]} cm
    - **Hand:** {stats_p1["hand"]}
    - **Recent winrate:** {stats_p1["recent_winrate"]*100:.1f}%
    - **H2H winrate vs {player2}:** {calculate_h2h_winrate(player1, player2)*100:.1f}%
    """)

with col2:
    st.subheader(player2)
    st.markdown(f"""
    - **Rank:** {stats_p2["rank"]}
    - **Age:** {int(stats_p2["age"])}
    - **Height:** {stats_p2["height"]} cm
    - **Hand:** {stats_p2["hand"]}
    - **Recent winrate:** {stats_p2["recent_winrate"]*100:.1f}%
    - **H2H winrate vs {player1}:** {calculate_h2h_winrate(player2, player1)*100:.1f}%
    """)

# -------------------------------
# Predict win probability
# -------------------------------
if st.button("🎾 Predict Match Outcome"):
    # 改成解包，predict_win_probability 返回 (p1_prob, p2_prob)
    prob_p1, prob_p2 = predict_win_probability(player1, player2)

    color_p1 = "#4CAF50" if prob_p1 >= prob_p2 else "#FF4F81"
    color_p2 = "#4CAF50" if prob_p2 >= prob_p1 else "#FF4F81"

    with col1:
        st.markdown(f"<h2 style='color:{color_p1}; font-size:28px;'>{prob_p1*100:.1f}%</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<h2 style='color:{color_p2}; font-size:28px;'>{prob_p2*100:.1f}%</h2>", unsafe_allow_html=True)

    # -------------------------------
    # -------------------------------
    h2h_matches = df[
        ((df["winner_name"] == player1) & (df["loser_name"] == player2)) |
        ((df["winner_name"] == player2) & (df["loser_name"] == player1))
    ].sort_values("tourney_date", ascending=False).head(5)

    if not h2h_matches.empty:
        st.subheader(f"🕑 Last {len(h2h_matches)} H2H Matches")
        h2h_matches['tourney_date'] = pd.to_datetime(h2h_matches['tourney_date'], format='%Y%m%d', errors='coerce')
        table = h2h_matches[['tourney_date', 'tourney_name', 'winner_name', 'loser_name', 'score']].copy()
        table.columns = ["Date", "Tournament", "Winner", "Loser", "Score"]
        table['Date'] = table['Date'].dt.strftime('%Y-%m-%d')
        st.table(table)
    else:
        st.info("No historical H2H matches found.")
