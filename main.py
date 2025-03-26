from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  
import joblib
from pydantic import BaseModel
import os

# Print the current working directory
print("Current working directory:", os.getcwd())


app = FastAPI(title="FitVerse BMI Classifier")

scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
model = load_model("models/sgd_momentum_model.h5")

class BMIPredictionInput(BaseModel):
    Weight: float
    Height: float
    BMI: float
    Age: float
    Gender: str

@app.get("/")
def read_root():
    return {"message": "Welcome to FitVerse BMI Classifier"}

@app.post("/predict")
async def predict_bmi(data: BMIPredictionInput):
    try:
        input_data = pd.DataFrame([[data.Weight, data.Height, data.BMI, data.Age, data.Gender]], 
                                  columns=["Weight", "Height", "BMI", "Age", "Gender"])
        input_data["BMI_to_Weight"] = input_data["BMI"] / input_data["Weight"]
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        input_data["Gender"] = label_encoders["Gender"].transform(input_data["Gender"])
        X_input = input_data.drop(columns=["Gender"])
        pred = model.predict(X_input)
        pred_class = label_encoders["BMIcase"].inverse_transform([np.argmax(pred)])[0]
        return {"predicted_bmi_case": pred_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    try:
        new_data = pd.read_csv(file.file)
        new_data.drop(columns=["Body Fat Percentage", "BFPcase", "Exercise Recommendation Plan"], inplace=True)
        new_data["BMI_to_Weight"] = new_data["BMI"] / new_data["Weight"]
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        new_data["Gender"] = label_encoders["Gender"].transform(new_data["Gender"])
        X_new = new_data.drop(columns=["BMIcase"])
        y_new = label_encoders["BMIcase"].transform(new_data["BMIcase"])
        y_new = to_categorical(y_new)  # Changed to match Colab
        model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)
        model.save("models/sgd_momentum_model.h5")
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
