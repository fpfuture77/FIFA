import pandas as pd
import json
import joblib

# Load config
with open("config.json", 'r') as f:
    config = json.load(f)

# Load new dataset
df = pd.read_csv("new_dataset.csv")

# Make predictions
predictions = {}
for position, features in config["positions"].items():
    X = df[features]

    # Load models
    tabnet = joblib.load(f"models/{position}_tabnet.pkl")
    rf = joblib.load(f"models/{position}_rf.pkl")
    xgb = joblib.load(f"models/{position}_xgb.pkl")

    # Ensemble prediction
    final_pred = (tabnet.predict_proba(X.values)[:, 1] + rf.predict_proba(X)[:, 1] + xgb.predict_proba(X)[:, 1]) / 3
    predictions[position] = df.iloc[final_pred.argsort()[-3:]].to_dict(orient='records')

print("✅ Predictions saved to predictions.json.")
