from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict

class TripData(BaseModel):
    trip_distance: float
    RatecodeID: int
    duration: float
    day_of_the_week: int
    pick_up_month: int
    is_weekend: bool
    is_rush_hour: bool
    is_late_night: bool
    boroughPickUpPoint: str
    boroughDropOffPoint: str
    is_airport_pickup: bool
    is_airport_dropoff: bool
    SameZone: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trip_distance": 7,
                "RatecodeID": 1,
                "duration": 12,
                "day_of_the_week": 3,
                "pick_up_month": 1,
                "is_weekend": False,
                "is_rush_hour": True,
                "is_late_night": False,
                "boroughPickUpPoint": "manhattan",
                "boroughDropOffPoint": "manhattan",
                "is_airport_pickup": True,
                "is_airport_dropoff": False,
                "SameZone": True
            }
        }
    )


app = FastAPI(
    title="TLC Fare-price Predictor",
    description="Predicts Fare-price for the taxi dynamicaly based on factors like rush hours , weeknd , ...",
    version="1.0.0"
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(base_dir, "..", "Models", "random_forest_model.pkl"))
print(model_path)


from preprocessing import cast_taxi_dtypes
import sys

# Make pickle able to find it, whether run via `python` or `uvicorn`
sys.modules["__main__"].cast_taxi_dtypes = cast_taxi_dtypes

model = joblib.load(model_path)



@app.post("/predict")
def predict_fare(trip: TripData):
    input_df = pd.DataFrame([trip.model_dump()])

    prediction = model.predict(input_df)[0]
    return {"predicted_fare": round(float(prediction), 2)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)