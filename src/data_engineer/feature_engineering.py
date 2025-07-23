import os
import pandas as pd
from datetime import datetime

# Define input and output paths
data_folder = os.path.join('../..', 'data', 'atp1_2000-2024')
output_path = os.path.join('../..', 'src', 'output','feature_dataset.csv')

# Load all match data
all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)

# Drop rows missing essential information
df = df.dropna(subset=['winner_name', 'loser_name', 'winner_rank', 'loser_rank'])

# Convert date
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['tourney_date'])

# Basic feature engineering
df['ranking_diff'] = df['loser_rank'] - df['winner_rank']
df['rank_points_diff'] = df['winner_rank_points'] - df['loser_rank_points']
df['age_diff'] = df['winner_age'] - df['loser_age']
df['height_diff'] = df['winner_ht'] - df['loser_ht']
df['same_hand'] = (df['winner_hand'] == df['loser_hand']).astype(int)
df['hand_matchup'] = df['winner_hand'].fillna('U') + '_' + df['loser_hand'].fillna('U')
df['label'] = 1  # will be used later

# ----------------------------------------------------------
# 🧠 Construct h2h_winrate: head-to-head winrate before match
# ----------------------------------------------------------
# Create match record with consistent player1/player2 order
h2h_records = []

for idx, row in df.iterrows():
    player1 = min(row['winner_name'], row['loser_name'])
    player2 = max(row['winner_name'], row['loser_name'])
    winner = row['winner_name']
    date = row['tourney_date']

    result = 1 if winner == player1 else 0  # 1 = player1 wins
    h2h_records.append([player1, player2, date, result])

h2h_df = pd.DataFrame(h2h_records, columns=['p1', 'p2', 'tourney_date', 'p1_win'])

# Sort by player pair and date
h2h_df = h2h_df.sort_values(['p1', 'p2', 'tourney_date'])

# Compute rolling winrate (no leakage)
h2h_df['p1_winrate'] = h2h_df.groupby(['p1', 'p2'])['p1_win'].transform(
    lambda x: x.shift().expanding().mean()
)

# Merge back to main df
def get_h2h_key(row):
    p1 = min(row['winner_name'], row['loser_name'])
    p2 = max(row['winner_name'], row['loser_name'])
    return pd.Series([p1, p2])

df[['p1', 'p2']] = df.apply(get_h2h_key, axis=1)

df = df.merge(
    h2h_df[['p1', 'p2', 'tourney_date', 'p1_winrate']],
    on=['p1', 'p2', 'tourney_date'],
    how='left'
)

# Convert to final h2h_winrate based on match direction
def assign_winrate(row):
    if row['winner_name'] == row['p1']:
        return row['p1_winrate']
    else:
        return 1 - row['p1_winrate'] if pd.notnull(row['p1_winrate']) else None

df['h2h_winrate'] = df.apply(assign_winrate, axis=1)

# Drop helper columns
df = df.drop(columns=['p1', 'p2', 'p1_winrate'])

# Fill missing h2h values with 0.5 (neutral)
df['h2h_winrate'] = df['h2h_winrate'].fillna(0.5)

# ----------------------------------------------------------
# 🧠 Construct recent_winrate: player recent form (past 5 matches)
# ----------------------------------------------------------

# Stack winner and loser into one long dataframe
winner_df = df[['tourney_date', 'winner_name']].copy()
winner_df['player'] = winner_df['winner_name']
winner_df['win'] = 1

loser_df = df[['tourney_date', 'loser_name']].copy()
loser_df['player'] = loser_df['loser_name']
loser_df['win'] = 0

winner_df = winner_df[['tourney_date', 'player', 'win']]
loser_df = loser_df[['tourney_date', 'player', 'win']]

all_matches = pd.concat([winner_df, loser_df], ignore_index=True)
all_matches = all_matches.sort_values(['player', 'tourney_date'])

# Compute rolling winrate (no leakage): last 5 matches
all_matches['recent_winrate'] = all_matches.groupby('player')['win'].transform(
    lambda x: x.shift().rolling(window=5, min_periods=1).mean()
)

# Merge recent_winrate back to original df
# First get winner and loser recent winrate
df = df.merge(
    all_matches[['tourney_date', 'player', 'recent_winrate']].rename(
        columns={'player': 'winner_name', 'recent_winrate': 'winner_recent_winrate'}
    ),
    on=['tourney_date', 'winner_name'],
    how='left'
)

df = df.merge(
    all_matches[['tourney_date', 'player', 'recent_winrate']].rename(
        columns={'player': 'loser_name', 'recent_winrate': 'loser_recent_winrate'}
    ),
    on=['tourney_date', 'loser_name'],
    how='left'
)

# Fill missing recent winrate with 0.5 (neutral)
df['winner_recent_winrate'] = df['winner_recent_winrate'].fillna(0.5)
df['loser_recent_winrate'] = df['loser_recent_winrate'].fillna(0.5)

# Save cleaned dataset
df.to_csv(output_path, index=False)
print(f"✅ Saved cleaned dataset with h2h to {output_path}")
