import streamlit as st
import pandas as pd
from urllib.parse import quote_plus

st.set_page_config(layout="wide")
st.title("Welcome to ATP Analyze")

df = pd.read_csv("src/output/player_dataset.csv")
players_2024 = df[df['tourney_date'].astype(str).str.startswith('2024')]
unique_players = sorted(set(players_2024['winner_name']).union(set(players_2024['loser_name'])))

search_name = st.text_input("🔍 Search Player by Name:")
if search_name:
    matched = [p for p in unique_players if search_name.lower() in p.lower()]
    if matched:
        for name in matched:
            player_link = f"/player_dashboard?player={quote_plus(name)}"
            st.markdown(f"- <a href='{player_link}' style='text-decoration:none;'>{name}</a>", unsafe_allow_html=True)
    else:
        st.warning("No player found with that name.")

st.markdown("---")

# Top5
st.subheader("🏆 Top 5 Players of 2024 (by Matches Won)")
top_winners = players_2024['winner_name'].value_counts().head(5)
for name, wins in top_winners.items():
    player_link = f"/player_dashboard?player={quote_plus(name)}"
    st.markdown(f"<a href='{player_link}' style='display:block; margin:4px 0;'><button>{name} ({wins} wins)</button></a>", unsafe_allow_html=True)
