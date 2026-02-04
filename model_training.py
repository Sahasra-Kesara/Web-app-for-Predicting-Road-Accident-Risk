import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
train_df = pd.read_csv("train.csv")

# Split X and y
X = train_df.drop(["accident_risk"], axis=1)
y = train_df["accident_risk"]

# Feature columns
categorical_cols = ["road_type", "lighting", "weather", "time_of_day"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Train/Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", rf_model)
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Validation RMSE
y_pred = pipeline.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print("Validation RMSE:", rmse)

# Save trained model
joblib.dump(pipeline, "model.pkl")

print("Model Saved Successfully as model.pkl")
