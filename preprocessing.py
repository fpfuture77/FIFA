import pandas as pd
import json
import os
from imblearn.over_sampling import SMOTE

# Load config
config_path = "C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Load dataset
df = pd.read_csv(config["dataset_path"])

# Clean column names to avoid mismatches (remove spaces, lowercase all names)
df.columns = df.columns.str.strip().str.replace(" ", "").str.lower()

# Define correct column name mappings (to fix mismatches dynamically)
column_mapping = {
    "gkdiving": "gk_diving",
    "gkhandling": "gk_handling",
    "gkkicking": "gk_kicking",
    "gkpositioning": "gk_positioning",
    "gkreflexes": "gk_reflexes"
}

# Rename columns dynamically based on mapping
df.rename(columns=column_mapping, inplace=True)

# Define relevant features dynamically (ensuring correct names)
required_features = [
    "overall", "composure", "strength", "aggression", "acceleration", "sprintspeed",
    "agility", "balance", "reactions", "stamina",
    "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
    "interceptions", "headingaccuracy", "marking", "standingtackle", "slidingtackle",
    "vision", "shortpassing", "longpassing", "ballcontrol", "dribbling",
    "finishing", "positioning", "shotpower", "volleys"
]

# Detect the correct position column dynamically
position_column = [col for col in df.columns if "position" in col.lower()]
if position_column:
    target_column = position_column[0]  # Use the detected column name
    required_features.append(target_column)
    config["target_column"] = target_column  # Update config
else:
    raise ValueError("Error: Could not find a 'Position' column in the dataset.")

# Filter only available columns (ignoring missing ones)
available_features = [col for col in required_features if col in df.columns]

# Log missing columns
missing_columns = set(required_features) - set(available_features)
if missing_columns:
    print(f"⚠️ Warning: The following columns were not found and will be skipped: {missing_columns}")

# Keep only selected numerical features
df = df[available_features]

# Handle missing values
df.fillna(df.median(), inplace=True)

# Apply SMOTE to balance classes
X = df.drop(columns=[config["target_column"]])
y = df[config["target_column"]]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save processed dataset
processed_path = "C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\FIFA-2019-processed.csv"
processed_df = pd.DataFrame(X_resampled, columns=X.columns)
processed_df[config["target_column"]] = y_resampled
processed_df.to_csv(processed_path, index=False)

# Save detected features for training
features_path = "C:\\Users\\Vinayak\\OneDrive\\Desktop\\FIFA_Model_Project\\features.json"
with open(features_path, 'w') as f:
    json.dump({"features": X.columns.tolist()}, f, indent=4)

print(f"✅ Preprocessing complete. Saved dataset as {processed_path}")
print(f"✅ Features detected and saved in {features_path}")
