import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/XGBoost_Final.pkl")

st.title("Insurance Charges Estimator")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
region = st.selectbox("region?", ["southwest", "southeast", "northwest", "southwest"])
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker?", ["No", "Yes"])
sex = st.selectbox("sex?", ["female", "male"])

if st.button("Estimate Charges"):
    user_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'sex': [sex],
        'region': [region]
        })
    charges = np.expm1(model.predict(user_data))
    st.success(f"Estimated charges: {charges[0]:.2f}")
