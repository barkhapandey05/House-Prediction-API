import streamlit as st
import requests

# API base URL (adjust if deployed)
API_URL = "http://127.0.0.1:8000"

st.title("🏠 California House Price Prediction")
st.write("Enter house features below to get a predicted price.")

# Input fields
MedInc = st.number_input("Median Income of Neighbourhood", min_value=0.0, step=0.1)
HouseAge = st.number_input("Median House Age", min_value=0.0, step=1.0)
AveRooms = st.number_input("Average Number of Rooms", min_value=0.0, step=0.1)
AveBedrms = st.number_input("Average Number of Bedrooms", min_value=0.0, step=0.1)
Population = st.number_input("Population of Block", min_value=0.0, step=1.0)
AveOccup = st.number_input("Average Household Members", min_value=0.0, step=0.1)
Latitude = st.slider("Latitude", min_value=32.0, max_value=42.0, step=0.01)
Longitude = st.slider("Longitude", min_value=-125.0, max_value=-114.0, step=0.01)

if st.button("Predict Price"):
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"🏡 Predicted Price: {result['Predicted Price']}")
            st.write(f"Short Format: {result['predicted_price_short']}")
            st.write(f"Confidence Range: {result['confidence_range']}")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Request failed: {e}")

# Batch prediction via CSV upload
st.subheader("📂 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    try:
        response = requests.post(f"{API_URL}/predict-file", files=files)
        if response.status_code == 200:
            st.download_button(
                label="Download Predictions CSV",
                data=response.content,
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Request failed: {e}")
