import pandas as pd
import json
import joblib
from sklearn.metrics import accuracy_score

# Load config
with open("C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\config.json", 'r') as f:
    config = json.load(f)

# Load dataset
df = pd.read_csv("C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\FIFA-2019-processed.csv")

# Evaluate models
results = {}
for position, features in config["positions"].items():
    X = df[features]
    y = (df[config["target_column"]] == position).astype(int)

    # Load models
    tabnet = joblib.load(f"models/{position}_tabnet.pkl")
    rf = joblib.load(f"models/{position}_rf.pkl")
    xgb = joblib.load(f"models/{position}_xgb.pkl")

    # Predict
    results[position] = {
        "TabNet Accuracy": accuracy_score(y, tabnet.predict(X.values)),
        "Random Forest Accuracy": accuracy_score(y, rf.predict(X)),
        "XGBoost Accuracy": accuracy_score(y, xgb.predict(X))
    }

# Save results
with open("results.json", 'w') as f:
    json.dump(results, f, indent=4)

print("✅ Evaluation complete. Results saved to results.json.")
