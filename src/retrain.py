import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import joblib
import os

# Global variables
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
model = load_model("models/sgd_momentum_model.h5")

def retrain_model(file):
    try:
        # Read uploaded CSV
        new_data = pd.read_csv(file)
        
        # Preprocess new data (same as in preprocessing.py)
        new_data.drop(columns=["Body Fat Percentage", "BFPcase", "Exercise Recommendation Plan"], inplace=True)
        new_data["BMI_to_Weight"] = new_data["BMI"] / new_data["Weight"]
        numerical_features = ["Weight", "Height", "BMI", "Age", "BMI_to_Weight"]
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        new_data["Gender"] = label_encoders["Gender"].transform(new_data["Gender"])
        X_new = new_data.drop(columns=["BMIcase"])
        y_new = label_encoders["BMIcase"].transform(new_data["BMIcase"])
        y_new = to_categorical(y_new)  # Match Colab's one-hot encoding
        
        # Retrain
        model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)
        model.save("models/sgd_momentum_model.h5")
        
        return "Model retrained successfully"
    except Exception as e:
        raise Exception(f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    print("This script is meant to be imported or run with a file-like object, not directly from command line.")
