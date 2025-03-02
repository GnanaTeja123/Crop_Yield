import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_data.csv")

# One-Hot Encode Categorical Columns
df = pd.get_dummies(df, drop_first=True)

# Features & Target Variable
X = df.drop(columns=["Yield"])  # Features
y = df["Yield"]  # Target

# Save feature names to ensure consistency
feature_names = X.columns.tolist()
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model Trained! MAE: {mae:.2f}, R²: {r2:.2f}")

# Save trained model
with open("models/crop_yield_model.pkl", "wb") as f:
    pickle.dump(model, f)
