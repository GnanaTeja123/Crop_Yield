import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("models/crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names used during training
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load dataset to fetch unique areas
df = pd.read_csv("data/processed/cleaned_data.csv")

# Streamlit UI
st.title("🌾 Crop Yield Prediction")

# User Inputs
area = st.selectbox("Select Region", df["Area"].unique())  # Dynamically fetch areas
year = st.number_input("Enter Year", min_value=1980, max_value=2030, step=1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)
pesticide_use = st.number_input("Pesticide Use (tonnes)", min_value=0.0, max_value=1000.0)

# Create a dictionary for input
input_dict = {"Year": year, "Rainfall": rainfall, "Temperature": temperature, "Pesticide_Use": pesticide_use}

# One-Hot Encode the selected area
area_encoded = pd.get_dummies(pd.DataFrame({"Area": [area]}), drop_first=True)

# Convert dictionary to DataFrame
input_df = pd.DataFrame([input_dict])

# Merge with one-hot encoded area columns
input_df = pd.concat([input_df, area_encoded], axis=1)

# Ensure all feature names match (add missing columns as 0)
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # Fill missing columns

# Drop any extra columns not in the trained model
input_df = input_df[feature_names]

# Predict Button
if st.button("Predict Yield"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Estimated Crop Yield: {prediction:.2f} hg/ha")
