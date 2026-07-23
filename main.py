import joblib
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

# Load the trained model and features
model = joblib.load("house_model.joblib")
feature_names = joblib.load("house_features.joblib")

#input schema for the API
class HouseFeatures(BaseModel):
    MedInc:    float = Field(gt=0, description="Median income of Neighbourhood")
    HouseAge:  float = Field(gt=0, description="Median house age of house in block")
    AveRooms:  float = Field(gt=0, description="Average number of rooms in house")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms in house")
    Population:float = Field(gt=0, description="Population of block")
    AveOccup:  float = Field(gt=0, description="Average number of household members")
    Latitude:  float = Field(ge=32, le=42, description="Latitude of block")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude of block")
    
#home
@app.get("/")
def home():
    return {
        "message": "Welcome to the California Housing Prediction API!",
        "status" : "running",
        "endpoints": "send POST requests to /predict"
    }

@app.get("/health")
def health():
    return {
        "Status": "Running",
        "model" : "Random Forest Regressor",
        "features" : feature_names,
        "avg_error" : "39,000 USD"
    }
    
#predictions
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([
            {
                "MedInc": house.MedInc,
                "HouseAge": house.HouseAge,
                "AveRooms": house.AveRooms,
                "AveBedrms": house.AveBedrms,
                "Population": house.Population,
                "AveOccup": house.AveOccup,
                "Latitude": house.Latitude,
                "Longitude": house.Longitude
            }
        ])
        
        # Make predictions using the trained model
        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000  # Convert to USD
        
        # Return the prediction as a JSON response
        return {
            "Predicted Price": f"{price_usd:,.0f}",
            "predicted_price_short" : f"{predicted:,.2f} hundred thousand USD",
            "confidence_range" : f"{price_usd - 39000:,.0f} USD to - {price_usd + 39000:,.0f} USD"
        }  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed {str(e)}")
    

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code = 400,
            detail="Invalid. Please upload a CSV file."
        )
    content = await file.read()
    
    df = pd.DataFrame(io.BytesIO(content))
    
    required_columns = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    missing_columns = [
        col for col in required_columns 
        if col not in df.columns
    ]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}"
        )
        
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty."
        )
    try:
        predictions = model.predict(df[required_columns])
        df["predicted_columns_usd"] = df["predicted_columns_usd"].apply(lambda x: f"{x :,.0f} USD")
        
        output = df.to_csv(index=False)
        
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
