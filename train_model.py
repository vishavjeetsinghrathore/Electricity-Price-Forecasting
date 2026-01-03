# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

# ===============================
# 2. Time-based Features
# ===============================
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["dayofweek"] = df["Datetime"].dt.dayofweek

# ===============================
# 3. Sliding Window (Lag Features)
# ===============================
def create_lag_features(data, lags=3):
    for lag in range(1, lags + 1):
        data[f"lag_{lag}"] = data["AEP_MW"].shift(lag)
    return data

df = create_lag_features(df, lags=3)
df.dropna(inplace=True)

# ===============================
# 4. Split Features & Target
# ===============================
X = df.drop(["Datetime", "AEP_MW"], axis=1)
y = df["AEP_MW"]

# ===============================
# 5. TimeSeries Cross Validation
# ===============================
tscv = TimeSeriesSplit(n_splits=5)

# ===============================
# 6. Models to Compare
# ===============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

# ===============================
# 7. Model Comparison
# ===============================
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring="r2"
    )

    results.append({
        "Model": name,
        "CV_R2_Mean": np.mean(cv_scores)
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison (TimeSeries CV):")
print(results_df)

# ===============================
# 8. Select Best Model
# ===============================
best_model_name = results_df.sort_values(
    "CV_R2_Mean", ascending=False
).iloc[0]["Model"]

print(f"\nBest Model Selected: {best_model_name}")

best_model = models[best_model_name]

# ===============================
# 9. Hyperparameter Tuning (ONLY BEST MODEL)
# ===============================
param_grids = {
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20]
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5]
    }
}

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_model)
])

if best_model_name in param_grids:
    print("\nApplying Hyperparameter Tuning...")

    grid_search = GridSearchCV(
        final_pipeline,
        param_grids[best_model_name],
        cv=tscv,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X, y)
    final_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV R2:", grid_search.best_score_)
else:
    final_pipeline.fit(X, y)
    final_model = final_pipeline

# ===============================
# 10. Save Final Model
# ===============================
joblib.dump(final_model, "best_model.pkl")
print("\nFinal model saved as best_model.pkl")
