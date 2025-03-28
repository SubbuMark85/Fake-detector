import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample dataset
data = pd.DataFrame({
    'account_age_days': [10, 1000, 200, 5, 800],
    'followers': [10, 5000, 300, 2, 4000],
    'following': [500, 200, 150, 1000, 100],
    'posts_per_day': [0.5, 1.2, 0.8, 10.0, 1.1],
    'is_fake': [1, 0, 0, 1, 0]
})

# Feature engineering
data['follower_following_ratio'] = data['followers'] / (data['following'] + 1e-5)

X = data[['account_age_days', 'posts_per_day', 'follower_following_ratio']]
y = data['is_fake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fake_profile_model.pkl')
print("Model saved!")
