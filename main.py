import tensorflow as tf

# Enable eager execution globally
tf.compat.v1.enable_eager_execution()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  
import joblib
from pydantic import BaseModel, root_validator, ValidationError
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print the current working directory
print("Current working directory:", os.getcwd())

app = FastAPI(title="FitVerse BMI Classifier")

# Loading the scaler, label encoders and model
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
model = load_model("models/sgd_momentum_model.h5")

class BMIPredictionInput(BaseModel):
    Weight: float
    Height: float
    BMI: float
    Age: float
    Gender: str

    @root_validator(pre=True)
    def check_valid_data(cls, values):
        weight = values.get('Weight')
        height = values.get('Height')
        bmi = values.get('BMI')
        age = values.get('Age')
        gender = values.get('Gender')

        # Validation checks
        if weight <= 0 or height <= 0 or bmi <= 0 or age <= 0:
            raise ValueError("All numeric values must be positive")
        if gender not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")

        return values

@app.get("/")
def read_root():
    return {"message": "Welcome to FitVerse BMIcase Classifier"}

@app.post("/predict")
async def predict_bmi(data: BMIPredictionInput):
    try:
        # Prepare input data
        input_data = pd.DataFrame([[data.Weight, data.Height, data.BMI, data.Age, data.Gender]], 
                                  columns=["Weight", "Height", "BMI", "Age", "Gender"])

        # Add BMI_to_Weight feature
        input_data["BMI_to_Weight"] = input_data["BMI"] / input_data["Weight"]

        # Define numerical features for scaling (including Gender)
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]
        
        # Scale the numerical features
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Encode the 'Gender' feature and append it to the numerical features
        input_data["Gender"] = label_encoders["Gender"].transform(input_data["Gender"])

        # Prepare the final input data for prediction
        final_input = input_data[["Weight", "Height", "BMI", "Age", "BMI_to_Weight", "Gender"]]

        # Predict using the model
        pred = model.predict(final_input)
        pred_class = label_encoders["BMIcase"].inverse_transform([np.argmax(pred)])[0]

        return {"predicted_bmi_case": pred_class}

    except ValidationError as e:
        logger.error(f"Input validation error: {e}")
        raise HTTPException(status_code=422, detail="Invalid input data")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing the prediction")


@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    try:
        # Enable eager execution 
        tf.compat.v1.enable_eager_execution()

        # Load the CSV file
        new_data = pd.read_csv(file.file)

        # Validate required columns exist
        required_columns = ["Weight", "Height", "BMI", "Age", "BMIcase", "Gender"]
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Preprocess the data
        new_data.drop(columns=["Body Fat Percentage", "BFPcase", "Exercise Recommendation Plan"], inplace=True, errors='ignore')
        new_data["BMI_to_Weight"] = new_data["BMI"] / new_data["Weight"]

        # Define numerical features for scaling
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]

        # Scale the features
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])

        # Encode the gender feature
        new_data["Gender"] = label_encoders["Gender"].transform(new_data["Gender"])

        # Prepare input and target variables
        X_new = new_data.drop(columns=["BMIcase"])
        y_new = label_encoders["BMIcase"].transform(new_data["BMIcase"])
        y_new = to_categorical(y_new)

        # Manually compile the model (if needed)
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        # Check input shape for compatibility
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Input data shape: {X_new.shape}")

        # Retrain the model
        model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)

        # Save the retrained model
        model.save("models/sgd_momentum_model.h5")
        
        logger.info("Model retrained and saved successfully.")
        return {"message": "Model retrained successfully"}

    except ValueError as e:
        logger.error(f"Validation error during retraining: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retraining the model")

