# Crop_Yield
Crop Yield Prediction
This project predicts crop yield using Machine Learning (ML) based on factors like rainfall, temperature, pesticide use, and region. The model is trained on real-world agricultural data and deployed using Streamlit for an interactive UI.

Folder Structure
Crop_Yield_Prediction/
│── data/
│   ├── raw/                      # Raw datasets
│   ├── processed/                 # Processed and cleaned datasets
│   │   ├── cleaned_data.csv       # Final dataset used for training
│
│── models/
│   ├── crop_yield_model.pkl       # Trained ML model
│   ├── feature_names.pkl          # Saved feature names (for inference)
│
│── scripts/
│   ├── preprocess_data.py         # Script for data preprocessing
│   ├── train_model.py             # Script for training the ML model
│
│── app/
│   ├── app.py                     # Streamlit app for predictions
│
│── README.md                      # Project documentation
│── requirements.txt                # Required Python packages

The project uses the following datasets:
rainfall.csv – Annual rainfall data
temp.csv – Average temperature data
pesticides.csv – Pesticide usage data
yield.csv – Historical crop yield data

1)Run the script to clean and preprocess the data:
python scripts/preprocess_data.py
2)Train the Machine Learning Model
python scripts/train_model.py
This will save the trained model as models/crop_yield_model.pkl.
3)Run the Streamlit App
streamlit run app/app.py
