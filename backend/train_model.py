import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
df = pd.read_csv("data/crop_yield.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Features and Target
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# Categorical and Numerical Columns
categorical_features = ["Area", "Item"]

numerical_features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training
pipeline.fit(X_train, y_train)

# Prediction
predictions = pipeline.predict(X_test)

# Evaluation
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Save Model
joblib.dump(pipeline, "models/crop_yield_model.pkl")

print("Model saved successfully!")