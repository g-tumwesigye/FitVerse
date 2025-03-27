import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

# Enable eager execution globally
tf.compat.v1.enable_eager_execution()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import joblib
from pydantic import BaseModel, root_validator, ValidationError
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Current working directory:", os.getcwd())

app = FastAPI(title="FitVerse BMI Classifier")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this to specific origins if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the pre-trained model, scaler, and label encoders
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
        input_data = pd.DataFrame([[data.Weight, data.Height, data.BMI, data.Age, data.Gender]], 
                                  columns=["Weight", "Height", "BMI", "Age", "Gender"])
        input_data["BMI_to_Weight"] = input_data["BMI"] / input_data["Weight"]
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        input_data["Gender"] = label_encoders["Gender"].transform(input_data["Gender"])
        final_input = input_data[["Weight", "Height", "BMI", "Age", "BMI_to_Weight", "Gender"]]
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
        # Load the new dataset
        new_data = pd.read_csv(file.file)

        # Validate required columns
        required_columns = ["Weight", "Height", "BMI", "Age", "BMIcase", "Gender"]
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Preprocess the data
        new_data.drop(columns=["Body Fat Percentage", "BFPcase", "Exercise Recommendation Plan"], inplace=True, errors='ignore')
        new_data["BMI_to_Weight"] = new_data["BMI"] / new_data["Weight"]
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        new_data["Gender"] = label_encoders["Gender"].transform(new_data["Gender"])
        X_new = new_data.drop(columns=["BMIcase"])
        y_new = label_encoders["BMIcase"].transform(new_data["BMIcase"])
        y_new_cat = to_categorical(y_new)

        # Retrain the model
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_new, y_new_cat, epochs=10, batch_size=32, verbose=0, validation_split=0.2)

        # Save the retrained model
        model.save("models/sgd_momentum_model.h5")
        logger.info("Model retrained and saved successfully.")

        # Evaluate on the new data
        y_pred_probs = model.predict(X_new)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true = y_new

        # Compute metrics
        test_loss = model.evaluate(X_new, y_new_cat, verbose=0)[0]
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        roc_auc = roc_auc_score(y_new_cat, y_pred_probs, multi_class='ovr', average='weighted')

        # Generate confusion matrix plot
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Generate loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        loss_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Return the response with metrics and visualizations
        return {
            "message": "Model retrained successfully",
            "metrics": {
                "test_loss": test_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            },
            "confusion_matrix": cm_base64,
            "loss_plot": loss_base64
        }
    except ValueError as e:
        logger.error(f"Validation error during retraining: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retraining the model")
