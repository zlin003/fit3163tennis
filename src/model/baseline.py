import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the raw match data
df = pd.read_csv('../output/cleaned_dataset.csv')
# Drop rows missing winner or loser
df = df.dropna(subset=['winner_name', 'loser_name'])

# Create target: always 1 (since this row represents a winner beating a loser)
df['target'] = 1

# Create simple string-based feature for demonstration
df['match_pair'] = df['winner_name'].astype(str) + '_vs_' + df['loser_name'].astype(str)

# Encode as categorical manually
df['match_pair'] = df['match_pair'].astype('category')

# Dummy baseline: treat match pair as only feature
features = ['match_pair']
X = df[features]
y = df['target']

# Encode categories
X_encoded = X.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)

# For binary classification we need two classes — create dummy negative samples by flipping names
df_fake = df.copy()
df_fake['target'] = 0
df_fake['match_pair'] = df['loser_name'].astype(str) + '_vs_' + df['winner_name'].astype(str)
df_fake['match_pair'] = df_fake['match_pair'].astype('category')
X_fake = df_fake[features]
X_fake_encoded = X_fake.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)

# Combine real and fake
X_full = pd.concat([X_encoded, X_fake_encoded], ignore_index=True)
y_full = pd.concat([df['target'], df_fake['target']], ignore_index=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# Train CatBoost model
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    verbose=50
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
