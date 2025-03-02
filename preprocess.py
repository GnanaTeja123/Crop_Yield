import pandas as pd
import numpy as np

# Load datasets
rainfall = pd.read_csv("data/raw/rainfall.csv")
temperature = pd.read_csv("data/raw/temp.csv")
yield_data = pd.read_csv("data/raw/yield.csv")
pesticides = pd.read_csv("data/raw/pesticides.csv")

# Rename columns for consistency
rainfall.rename(columns={" Area": "Area", "Year": "Year", "average_rain_fall_mm_per_year": "Rainfall"}, inplace=True)
temperature.rename(columns={"year": "Year", "country": "Area", "avg_temp": "Temperature"}, inplace=True)
yield_data.rename(columns={"Area": "Area", "Year": "Year", "Value": "Yield"}, inplace=True)
pesticides.rename(columns={"Area": "Area", "Year": "Year", "Value": "Pesticide_Use"}, inplace=True)

# Convert data types
rainfall["Rainfall"] = pd.to_numeric(rainfall["Rainfall"], errors="coerce")
temperature["Temperature"] = pd.to_numeric(temperature["Temperature"], errors="coerce")
yield_data["Yield"] = pd.to_numeric(yield_data["Yield"], errors="coerce")
pesticides["Pesticide_Use"] = pd.to_numeric(pesticides["Pesticide_Use"], errors="coerce")

# Merge datasets on Area & Year
df = yield_data.merge(rainfall, on=["Area", "Year"], how="left")
df = df.merge(temperature, on=["Area", "Year"], how="left")
df = df.merge(pesticides, on=["Area", "Year"], how="left")

# Drop missing values
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("data/processed/cleaned_data.csv", index=False)
print("✅ Data Preprocessing Completed!")
