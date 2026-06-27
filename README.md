# 🏠 California Housing Prediction API

A FastAPI-powered machine learning API that predicts California house prices using a trained **Random Forest Regressor** model.  
It supports both **JSON requests** and **CSV file uploads** for batch predictions.

---

## 🚀 Features
- RESTful API built with **FastAPI**
- Predict house prices from structured input
- Batch predictions via CSV file upload
- Input validation with **Pydantic**
- Health check endpoint with model details
- JSON and CSV response formats

---

## 📂 Project Structure
house-prediction-api/
│── app.py                # FastAPI frontend application
│── main.py               # FastAPI application
│── train.py              # Training model
│── house_model.joblib    # Trained ML model
│── house_features.joblib # Feature names used by the model
│── requirements.txt      # Dependencies
│── README.md  


---

## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/your-username/house-prediction-api.git
   cd house-prediction-api

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python train.py

4. Run the FastAPI server:
   uvicorn app:app --reload

5. Run Streamlit UI:
   streamlit run streamlit_app.py

Open the browser at http://localhost:8501 to interact with your API.
