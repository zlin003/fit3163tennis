import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Load match data and convert date format
@st.cache_data
def load_data():
    df = pd.read_csv('src/output/player_dataset.csv')
    df['tourney_date'] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
    return df

df = load_data()

# App title
st.title('🎾 Tennis Player Career Dashboard')

# Count total wins and get list of players by win frequency
win_counts = df['winner_name'].value_counts()
player_list = list(win_counts.index)

# Search box to filter player list
search_input = st.text_input('🔍 Search player name:')
filtered_players = [name for name in player_list if search_input.lower() in name.lower()]
player_name = st.selectbox('Select a player from search results:', filtered_players)

# Main content after player is selected
if player_name:
    st.header(f"{player_name} - Career Summary")

    # Filter all matches of the player
    player_matches = df[(df['winner_name'] == player_name) | (df['loser_name'] == player_name)].copy()
    player_matches = player_matches.dropna(subset=['tourney_date'])
    player_matches['year'] = player_matches['tourney_date'].dt.year

    # Calculate basic stats
    total_matches = len(player_matches)
    total_wins = (player_matches['winner_name'] == player_name).sum()
    winrate = round(total_wins / total_matches * 100, 2) if total_matches else 0

    st.write(f"**Total Matches:** {total_matches}")
    st.write(f"**Total Wins:** {total_wins}")
    st.write(f"**Overall Winrate:** {winrate}%")

    # Winrate per year chart
    wins_per_year = player_matches[player_matches['winner_name'] == player_name].groupby('year').size()
    matches_per_year = player_matches.groupby('year').size()
    winrate_per_year = (wins_per_year / matches_per_year).fillna(0) * 100

    st.subheader('📈 Winrate by Year')
    fig, ax = plt.subplots()
    winrate_per_year.sort_index().plot(ax=ax, marker='o', linestyle='-')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('Winrate (%)')
    ax.set_xlabel('Year')
    ax.set_title(f'{player_name} - Annual Winrate')
    ax.set_xlim(player_matches['year'].min(), player_matches['year'].max())
    st.pyplot(fig)

    # Surface win distribution (pie chart)
    st.subheader('🎾 Surface Winrate Distribution')
    surface_matches = player_matches.dropna(subset=['surface'])
    surface_win_counts = surface_matches[surface_matches['winner_name'] == player_name]['surface'].value_counts()
    fig2, ax2 = plt.subplots()
    surface_win_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel('')
    ax2.set_title(f'{player_name} Wins by Surface')
    st.pyplot(fig2)

    # Show recent 20 matches, sorted with wins first
    st.subheader('📋 Recent 20 Matches (Wins First)')
    player_matches['win_flag'] = (player_matches['winner_name'] == player_name).astype(int)
    recent_matches_sorted = player_matches.sort_values(by=['win_flag', 'tourney_date'], ascending=[False, False])
    st.dataframe(recent_matches_sorted[['tourney_date', 'tourney_name', 'surface', 'winner_name', 'loser_name', 'score']].head(20))

    # Opponent analysis section
    st.subheader('🎯 Opponent Analysis')

    # Create opponent column
    player_matches['opponent'] = player_matches.apply(
        lambda row: row['loser_name'] if row['winner_name'] == player_name else row['winner_name'],
        axis=1
    )

    # Count matches and wins vs each opponent
    opponent_counts = player_matches['opponent'].value_counts()
    wins_vs_opponent = player_matches[player_matches['winner_name'] == player_name]['loser_name'].value_counts()

    # Combine into one dataframe
    opponent_df = pd.DataFrame({
        'Matches': opponent_counts,
        'Wins': wins_vs_opponent
    }).fillna(0)

    opponent_df['Winrate (%)'] = (opponent_df['Wins'] / opponent_df['Matches'] * 100).round(2)
    opponent_df = opponent_df.astype({'Matches': int, 'Wins': int})

    # Top 5 opponents by number of matches
    st.write('**Top 5 Most Played Opponents:**')
    st.dataframe(opponent_df.sort_values(by='Matches', ascending=False).head(5))

    # Top 5 toughest (lowest winrate, min 5 matches)
    tough_opponents = opponent_df[opponent_df['Matches'] >= 5].sort_values(by='Winrate (%)').head(5)
    st.write('**Top 5 Toughest Opponents (min 5 matches):**')
    st.dataframe(tough_opponents)

    # Top 5 easiest (highest winrate, min 5 matches)
    easy_opponents = opponent_df[opponent_df['Matches'] >= 5].sort_values(by='Winrate (%)', ascending=False).head(5)
    st.write('**Top 5 Easiest Opponents (min 5 matches):**')
    st.dataframe(easy_opponents)
