import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from urllib.parse import quote_plus

# Set wide layout
st.set_page_config(layout="wide")

# Hide default Streamlit header
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Custom top banner with title
# Hide default header
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Custom full-width top banner
st.markdown("""
    <style>
    .custom-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 80px;
        background-color: #0b162b;
        color: white;
        display: flex;
        align-items: center;
        padding-left: 2rem;
        z-index: 1000;
    }
    .main {
        padding-top: 100px; /* prevent content hiding behind fixed header */
    }
    </style>

    <div class="custom-header">
        <h1 style="color:white; font-size:26px; margin: 0;">🎾 ATP Analyze</h1>
    </div>
""", unsafe_allow_html=True)


# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('src/output/player_dataset.csv')
    player_static_df = pd.read_csv('data/atp_players.csv')
    df['tourney_date'] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
    return df, player_static_df

df, player_static_df = load_data()

# -------------------------------
# Get Player Input
# -------------------------------
win_counts = df['winner_name'].value_counts()
player_list = list(win_counts.index)

query_params = st.query_params
player_param = query_params.get("player", None)
player_name = None

if player_param:
    matched_players = [name for name in player_list if name.lower().strip() == player_param.lower().strip()]
    if matched_players:
        player_name = matched_players[0]
    else:
        st.warning(f"No exact match for player: {player_param}")

if not player_name:
    search_input = st.text_input('🔍 Enter player name (full or partial):')
    if search_input:
        matched_players = [name for name in player_list if search_input.lower() in name.lower()]
        if matched_players:
            player_name = matched_players[0]
            st.success(f"Showing results for: **{player_name}**")
        else:
            st.warning("No matching player found.")

# -------------------------------
# Show Player Summary
# -------------------------------
if player_name:
    st.header(f"{player_name} - Summary")

    player_matches = df[(df['winner_name'] == player_name) | (df['loser_name'] == player_name)].copy()
    player_matches = player_matches.dropna(subset=['tourney_date'])
    player_matches['year'] = player_matches['tourney_date'].dt.year

    total_matches = len(player_matches)
    total_wins = (player_matches['winner_name'] == player_name).sum()
    winrate = round(total_wins / total_matches * 100, 2) if total_matches else 0

    def get_static_info(player_name, static_df):
        try:
            first, last = player_name.split()
            match = static_df[
                (static_df['name_first'].str.lower() == first.lower()) &
                (static_df['name_last'].str.lower() == last.lower())
            ]
            return match.iloc[0] if not match.empty else None
        except:
            return None

    static_info = get_static_info(player_name, player_static_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Player Info")
        if static_info is not None:
            dob_raw = str(int(static_info['dob'])) if pd.notna(static_info['dob']) else ""
            dob_formatted = f"{dob_raw[:4]}/{dob_raw[4:6]}/{dob_raw[6:]}" if len(dob_raw) == 8 else "Unknown"
            hand_map = {'R': 'Right (R)', 'L': 'Left (L)', 'U': 'Unknown'}
            hand_str = hand_map.get(static_info['hand'], static_info['hand'])

            country_map = {
                'ESP': 'Spain', 'USA': 'United States', 'FRA': 'France', 'SUI': 'Switzerland', 'SRB': 'Serbia',
                'GBR': 'United Kingdom', 'GER': 'Germany', 'ARG': 'Argentina', 'AUS': 'Australia', 'RUS': 'Russia',
                'ITA': 'Italy', 'CRO': 'Croatia', 'CAN': 'Canada', 'SWE': 'Sweden', 'JPN': 'Japan',
                'ROU': 'Romania', 'CZE': 'Czech Republic', 'DEN': 'Denmark', 'HUN': 'Hungary', 'CHN': 'China'
            }
            country_full = country_map.get(static_info['ioc'], static_info['ioc'])

            st.markdown(f"""
            - **Playing Hand:** {hand_str}  
            - **Date of Birth:** {dob_formatted}  
            - **Height:** {static_info['height']} cm  
            - **Country:** {country_full}
            """)
        else:
            st.info("No static info available for this player.")

        st.markdown("---")
        st.subheader("Career Summary")
        st.markdown(f"**Total Matches:** {total_matches}")
        st.markdown(f"**Total Wins:** {total_wins}")
        st.markdown(f"**Overall Winrate:** {winrate}%")

    with col2:
        st.subheader("📈 Winrate by Year")
        if not player_matches.empty:
            wins_per_year = player_matches[player_matches['winner_name'] == player_name].groupby('year').size()
            matches_per_year = player_matches.groupby('year').size()
            winrate_per_year = (wins_per_year / matches_per_year).fillna(0) * 100
            fig, ax = plt.subplots()
            winrate_per_year.sort_index().plot(ax=ax, marker='o')
            ax.set_ylabel('Winrate (%)')
            ax.set_xlabel('Year')
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_xlim(player_matches['year'].min(), player_matches['year'].max())
            st.pyplot(fig)
        else:
            st.info("No yearly data available.")

    with col3:
        st.subheader("🎾 Surface Winrate Distribution")
        surface_matches = player_matches.dropna(subset=['surface'])
        surface_win_counts = surface_matches[surface_matches['winner_name'] == player_name]['surface'].value_counts()
        if not surface_win_counts.empty:
            fig2, ax2 = plt.subplots()
            surface_win_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
            ax2.set_ylabel('')
            st.pyplot(fig2)
        else:
            st.info("No surface data available.")

    st.subheader('📋 Recent 20 Matches (Wins First)')

    player_matches['win_flag'] = (player_matches['winner_name'] == player_name).astype(int)
    recent_matches_sorted = player_matches.sort_values(by=['win_flag', 'tourney_date'], ascending=[False, False])
    recent_matches_sorted['tourney_date'] = recent_matches_sorted['tourney_date'].dt.strftime('%Y-%m-%d')

    def make_link(name):
        return f"[{name}](?player={quote_plus(name.strip())})"

    table = recent_matches_sorted[['tourney_date', 'tourney_name', 'surface', 'winner_name', 'loser_name', 'score']].head(20).copy()
    table.columns = ['Date', 'Tournament', 'Surface', 'Winner', 'Loser', 'Score']
    table['Winner'] = table['Winner'].astype(str).apply(make_link)
    table['Loser'] = table['Loser'].astype(str).apply(make_link)

    st.markdown(table.to_markdown(index=False), unsafe_allow_html=True)

    st.subheader('🎯 Opponent Analysis')

    player_matches['opponent'] = player_matches.apply(
        lambda row: row['loser_name'] if row['winner_name'] == player_name else row['winner_name'],
        axis=1
    )

    opponent_counts = player_matches['opponent'].value_counts()
    wins_vs_opponent = player_matches[player_matches['winner_name'] == player_name]['loser_name'].value_counts()

    opponent_df = pd.DataFrame({
        'Matches': opponent_counts,
        'Wins': wins_vs_opponent
    }).fillna(0)
    opponent_df['Winrate (%)'] = (opponent_df['Wins'] / opponent_df['Matches'] * 100).round(2)
    opponent_df = opponent_df.astype({'Matches': int, 'Wins': int})

    st.write('**Top 5 Most Played Opponents:**')
    st.dataframe(opponent_df.sort_values(by='Matches', ascending=False).head(5))

    st.write('**Top 5 Toughest Opponents (min 5 matches):**')
    st.dataframe(opponent_df[opponent_df['Matches'] >= 5].sort_values(by='Winrate (%)').head(5))

    st.write('**Top 5 Easiest Opponents (min 5 matches):**')
    st.dataframe(opponent_df[opponent_df['Matches'] >= 5].sort_values(by='Winrate (%)', ascending=False).head(5))
