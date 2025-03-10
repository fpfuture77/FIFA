import pandas as pd
import json
import joblib
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load config
with open("C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\config.json", 'r') as f:
    config = json.load(f)

# Load processed data
df = pd.read_csv("C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\FIFA-2019-processed.csv")

# Create models directory
model_dir = "C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\models"
os.makedirs(model_dir, exist_ok=True)

# Train models per position
for position, features in config["positions"].items():
    X = df[features]
    y = (df[config["target_column"]] == position).astype(int)

    # Train models
    tabnet = TabNetClassifier().fit(X.values, y.values, max_epochs=100, patience=10)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X, y)

    # Save models
    joblib.dump(tabnet, f"{model_dir}\\{position}_tabnet.pkl")
    joblib.dump(rf, f"{model_dir}\\{position}_rf.pkl")
    joblib.dump(xgb, f"{model_dir}\\{position}_xgb.pkl")

    print(f"✅ Trained and saved models for {position}.")

print("✅ Training complete.")
