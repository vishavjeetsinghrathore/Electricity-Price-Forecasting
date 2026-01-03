# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
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
# 3. Lag Features
# ===============================
for lag in [1, 2, 3]:
    df[f"lag_{lag}"] = df["AEP_MW"].shift(lag)

df.dropna(inplace=True)

# ===============================
# 4. Features / Target
# ===============================
X = df.drop(["Datetime", "AEP_MW"], axis=1)
y = df["AEP_MW"]

# ===============================
# 5. TimeSeries CV
# ===============================
tscv = TimeSeriesSplit(n_splits=5)

# ===============================
# 6. Models (DEPLOYMENT SAFE)
# ===============================
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,          # ⬅ reduced
        max_depth=12,             # ⬅ limits tree size
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []

# ===============================
# 7. Model Comparison
# ===============================
for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=tscv,
        scoring="r2"
    )

    results.append({
        "Model": name,
        "CV_R2_Mean": scores.mean()
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison (TimeSeries CV)")
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
# 9. Light Hyperparameter Tuning
# ===============================
if best_model_name == "Random Forest":
    param_grid = {
        "n_estimators": [40, 60],
        "max_depth": [10, 12]
    }

    grid = GridSearchCV(
        best_model,
        param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X, y)
    final_model = grid.best_estimator_

    print("Best Params:", grid.best_params_)
    print("Best CV R2:", grid.best_score_)

else:
    final_model = best_model.fit(X, y)

# ===============================
# 10. Save Model (COMPRESSED)
# ===============================
joblib.dump(final_model, "best_model.pkl", compress=3)
print("\nModel saved as best_model.pkl")
