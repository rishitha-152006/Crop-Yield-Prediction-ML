import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("../dataset/crop_yield.csv")

# Encode categorical columns
area_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["Area"] = area_encoder.fit_transform(df["Area"])
df["Item"] = item_encoder.fit_transform(df["Item"])

# Features and Target
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, predictions)

print("=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)
print(f"R² Score: {r2:.4f}")

# Save model
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(area_encoder, "area_encoder.pkl")
joblib.dump(item_encoder, "item_encoder.pkl")

print("\nFiles Saved:")
print("- crop_yield_model.pkl")
print("- area_encoder.pkl")
print("- item_encoder.pkl")