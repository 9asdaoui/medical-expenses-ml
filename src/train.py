import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib



#import clean data
df = pd.read_csv("data/processed/cleaned_data.csv")
print(df)
#split data
X = df.drop(columns=['charges'])

y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9
)
print("data plited")
#normalize data
scaler = MinMaxScaler() 

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("data normalized")



# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "SVR": SVR()
}

print("models called")

results = {}
for name, model in models.items():
    print(name)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")  # save trained model

# Save results
pd.DataFrame(results).T.to_csv("reports/model_results.csv")