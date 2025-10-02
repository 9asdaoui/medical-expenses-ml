import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from processe import X_train, X_test, y_train, y_test
from sklearn.svm import SVR


# ========= Preprocessor =========
num_features = ["age", "bmi", "children"]
cat_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# ========= Define Models with Best Params =========
models = {
    "RandomForest_Final": RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        random_state=151
    ),
    "XGBoost_Final": XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=151,
        n_jobs=-1
    ),
    "SVR_Final": SVR(
        C= 10, 
        epsilon= 0.1, 
        kernel= "rbf"
    )
}

# ========= Train, Evaluate, Save =========
results = []
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\n🔹 Training {name} ...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} → Test MSE: {mse:.2f}, Test R2: {r2:.4f}")

    # Save model
    joblib.dump(pipeline, f"models/{name}.pkl")

    # Save results
    results.append({
        "Model": name,
        "Test MSE": mse,
        "Test R2": r2
    })

# ========= Save Report =========
report_file = "reports/final_train_results.csv"

new_results = pd.DataFrame(results)

new_results.to_csv(report_file, index=False)

print(f"\n📊 Results saved in {report_file}")
