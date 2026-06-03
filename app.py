import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/car_price_model.pkl")

st.title("Car Price Prediction")

year = st.number_input("Year", 2000, 2026)
present_price = st.number_input("Present Price")
driven_kms = st.number_input("Driven Kms")
owner = st.number_input("Owner", 0, 5)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Convert values
car_age = 2026 - year
fuel_type_Diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_Petrol = 1 if fuel_type == "Petrol" else 0
selling_type_Individual = 1 if selling_type == "Individual" else 0
transmission_Manual = 1 if transmission == "Manual" else 0

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Owner': [owner],
        'Car_Age': [car_age],
        'Fuel_Type_Diesel': [fuel_type_Diesel],
        'Fuel_Type_Petrol': [fuel_type_Petrol],
        'Selling_type_Individual': [selling_type_Individual],
        'Transmission_Manual': [transmission_Manual]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Car Price: ₹ {prediction[0]:.2f} Lakhs")