```markdown
# FitVerse BMI Classifier

FitVerse is a machine learning application that classifies Body Mass Index (BMI) cases based on user input, using a pre-trained model. The app also allows for model retraining with new data.

## Features

- **BMI Classification**: Predict the BMI case (e.g., underweight, normal weight, overweight, etc.) based on user input.
- **Model Retraining**: Retrain the model using new data uploaded through the `/retrain` endpoint.
- **FastAPI Backend**: The application is built with FastAPI for fast, asynchronous handling of requests.

## Requirements

The project requires the following dependencies:

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Uvicorn
- Pandas
- Numpy
- Joblib

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/FitVerse.git
   cd FitVerse
   ```
Create a virtual environment 

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have `requirements.txt` in your project with all necessary dependencies listed.

## Model Files

The model files (`scaler.pkl`, `label_encoders.pkl`, and `sgd_momentum_model.h5`) are required to make predictions. These files should be stored in the `models/` directory in your project.

- **scaler.pkl**: The fitted scaler for feature normalization.
- **label_encoders.pkl**: The fitted label encoders for encoding categorical variables.
- **sgd_momentum_model.h5**: The pre-trained Keras model.

## Running the Application

To run the FastAPI application locally:

1. Start the FastAPI server with Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

2. The server will run on `http://127.0.0.1:8000`. You can access the interactive API documentation at:

   ```
   http://127.0.0.1:8000/docs
   ```

   This allows you to interact with the `predict` and `retrain` endpoints.

## API Endpoints

### 1. **Predict BMI Case**

- **Endpoint**: `/predict`
- **Method**: POST
- **Request Body**: JSON object containing the following fields:
  
  ```json
  {
    "Weight": float,
    "Height": float,
    "BMI": float,
    "Age": float,
    "Gender": "Male" or "Female"
  }
  ```

- **Response**:

  ```json
  {
    "predicted_bmi_case": "BMI classification (e.g., 'Normal', 'Overweight', etc.)"
  }
  ```

### 2. **Retrain Model**

- **Endpoint**: `/retrain`
- **Method**: POST
- **Request Body**: CSV file containing the following columns:

  - `Weight`: User's weight in kg.
  - `Height`: User's height in cm.
  - `BMI`: User's BMI value.
  - `Age`: User's age in years.
  - `Gender`: User's gender ("Male" or "Female").
  - `BMIcase`: The classification of the BMI (e.g., "Underweight", "Normal weight", "Overweight", etc.).

- **Response**:

  ```json
  {
    "message": "Model retrained successfully"
  }
  ```

## Troubleshooting

- Ensure that the model files (`scaler.pkl`, `label_encoders.pkl`, and `sgd_momentum_model.h5`) are placed in the `models/` directory.
- If you encounter issues with the model's shape or prediction errors, check the input data format and ensure that features like BMI, height, weight, and gender are correctly scaled and encoded.




