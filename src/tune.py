import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor
from processe import X_train, X_test, y_train, y_test
from sklearn.svm import SVR


# ========= Preprocessor =========
num_features = ["age", "bmi", "children"]
cat_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# ========= Models & Params =========
models = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=151),
        "params": {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=151),
        "params": {
            "model__n_estimators": [100, 200, 500],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    },
    "SVR" : {
        "model": SVR(),
        "params":{
            "model__kernel": ["rbf", "linear"],
            "model__C": [0.1, 1, 10, 100],
            "model__epsilon": [0.01, 0.1, 0.5, 1]
        }
    }
}

# ========= Train & Save =========
results = []

for name, config in models.items():
    print(f"\n===== Training {name} =====")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", config["model"])
    ])
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["params"],
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"Best CV R2 for {name}: {grid_search.best_score_}")
    print(f"Test MSE for {name}: {mse}")
    print(f"Test R2 for {name}: {r2}")
    
    # Save best model
    joblib.dump(best_model, f"models/best_{name}.pkl")
    
    # Save results
    results.append({
        "Model": name,
        "Best Params": grid_search.best_params_,
        "CV R2": grid_search.best_score_,
        "Test MSE": mse,
        "Test R2": r2
    })

# ========= Save Report =========
report_df = pd.DataFrame(results)
report_df.to_csv("reports/tuning_results.csv", index=False)
print("\n✅ Training finished. Results saved to reports/tuning_results.csv")
