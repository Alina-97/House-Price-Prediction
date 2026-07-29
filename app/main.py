from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from fastapi import HTTPException

import joblib
import logging

app = FastAPI()
logging.basicConfig(
    level=logging.INFO
)

# 1. CURRENT_DIR points to the 'app' folder
CURRENT_DIR = Path(__file__).resolve().parent
# 2. Add .parent to step up out of 'app' into the project root
MODEL_FILE = CURRENT_DIR.parent / "models" / "house_price_model.pkl"


# Now it loads perfectly, permanently, without you thinking about it
model = joblib.load(MODEL_FILE)

# called pydantic in python (ur dto)to validate the request body
class HouseRequest(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(gt=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float
    Longitude: float
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

        return {
            "predicted_house_price":
            float(prediction[0])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )