import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from celsius_to_fahrenheit_ml import create_model, train_model, predict

app = FastAPI() # to test the API run on terminal: uvicorn main:app --reload
                # to comsume run oon terminal: curl -X POST "http://127.0.0.1:8000/convert" -H "Content-Type: application/json" -d '{"celsius": 32.0}'

class TemperatureRequest(BaseModel):
    celsius: float

class TemperatureResponse(BaseModel):
    fahrenheit: float

def main(celsius: float) -> float:
    
    celsius_q    = np.array([40, -10,  0,  8, 15, 22,  38],   dtype=float)
    fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
    
    model = create_model()
    model = train_model(model, celsius_q, fahrenheit_a)
    prediction_fahrenheit = predict(model, celsius)

    return prediction_fahrenheit


@app.post("/convert", response_model=TemperatureResponse)
async def convert_temperature(request: TemperatureRequest):
    prediction_fahrenheit = main(request.celsius)
    return TemperatureResponse(fahrenheit=prediction_fahrenheit)

# Rota opcional para testar com GET
@app.get("/convert/{celsius}", response_model=TemperatureResponse)
async def convert_temperature_get(celsius: float):
    result = main(celsius)
    return TemperatureResponse(fahrenheit=result)