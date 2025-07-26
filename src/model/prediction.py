import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# Load the cleaned match dataset
df = pd.read_csv('../output/feature_dataset.csv')

# Drop rows with missing values for key match features
df = df.dropna(subset=[
    'winner_rank', 'loser_rank',
    'winner_rank_points', 'loser_rank_points',
    'winner_age', 'loser_age',
    'winner_ht', 'loser_ht',
    'winner_hand', 'loser_hand'
])

# Create features based only on pre-match information
df['ranking_diff'] = df['loser_rank'] - df['winner_rank']
df['rank_points_diff'] = df['winner_rank_points'] - df['loser_rank_points']
df['age_diff'] = df['winner_age'] - df['loser_age']
df['height_diff'] = df['winner_ht'] - df['loser_ht']
df['same_hand'] = (df['winner_hand'] == df['loser_hand']).astype(int)
df['hand_matchup'] = df['winner_hand'].fillna('U') + '_' + df['loser_hand'].fillna('U')
df['recent_winrate_diff'] = df['winner_recent_winrate'] - df['loser_recent_winrate']


# Assign label = 1 for original matches
df['label'] = 1

# Create symmetric matches by flipping winner/loser data
flipped = df.copy()
flipped['ranking_diff'] = -flipped['ranking_diff']
flipped['rank_points_diff'] = -flipped['rank_points_diff']
flipped['age_diff'] = -flipped['age_diff']
flipped['height_diff'] = -flipped['height_diff']
flipped['same_hand'] = flipped['same_hand']
flipped['hand_matchup'] = flipped['hand_matchup'].apply(lambda x: '_'.join(reversed(x.split('_'))))
flipped['recent_winrate_diff'] = -flipped['recent_winrate_diff']

flipped['label'] = 0

# Combine original and flipped datasets
final_df = pd.concat([df, flipped], ignore_index=True)

# Define features and target
features = [
    'ranking_diff', 'rank_points_diff',
    'age_diff', 'height_diff',
    'same_hand', 'hand_matchup',
    'h2h_winrate', 'recent_winrate_diff'
]


target = 'label'
categorical_features = ['hand_matchup']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    final_df[features], final_df[target],
    test_size=0.2, random_state=42, stratify=final_df[target]
)

# Create CatBoost pools
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Train CatBoost model
# Final model training with best parameters
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.2717659141225842,
    depth=10,
    l2_leaf_reg=3.1545772618914834,
    random_strength=0.041159733940320326,
    bagging_temperature=0.994656919578217,
    border_count=110,
    loss_function='Logloss',
    eval_metric='Accuracy',
    early_stopping_rounds=30,
    random_seed=42,
    verbose=50,
    task_type='GPU'
)

model.fit(train_pool, eval_set=test_pool)

model.save_model("../model/catboost_model.cbm")
print("model saved to ../model/catboost_model.cbm")

# Evaluate model
y_pred = model.predict(X_test)
print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# --- SHAP Value Analysis ---
print("\n🔍 Running SHAP value analysis...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot feature importance (SHAP summary bar plot)
shap.summary_plot(shap_values, X_test, plot_type="bar")
