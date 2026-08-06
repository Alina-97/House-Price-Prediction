from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Path to model
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_FILE = CURRENT_DIR.parent / "models" / "house_price_model.pkl"

# Load model once at startup
model = joblib.load(MODEL_FILE)


# Request DTO
from pydantic import BaseModel, ConfigDict, Field

class HouseRequest(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(gt=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float
    Longitude: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 5.8412,
                "HouseAge": 34,
                "AveRooms": 6.1245,
                "AveBedrms": 1.0204,
                "Population": 2405,
                "AveOccup": 2.8715,
                "Latitude": 34.0522,
                "Longitude": -118.2437
            }
        }
    )

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: HouseRequest):

    try:

        features = [[
            request.MedInc,
            request.HouseAge,
            request.AveRooms,
            request.AveBedrms,
            request.Population,
            request.AveOccup,
            request.Latitude,
            request.Longitude
        ]]

        prediction = model.predict(features)

        logging.info(f"Prediction generated: {prediction[0]}")

        return {
            "predicted_house_price": float(prediction[0])
        }

    except Exception as e:

        logging.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )