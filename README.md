# FitVerse - BMI Case Classifier

## Project Description
FitVerse is a health monitoring web application that uses machine learning to predict BMI cases and provide personalized health insights. Users can input their weight, height, age, BMI, and gender to get a predicted BMI case (e.g., normal, overweight). Additionally, FitVerse allows users to retrain the underlying model with new datasets, displaying detailed performance metrics (Test Loss, Accuracy, Precision, Recall, F1 Score, ROC AUC) and visualizations (Confusion Matrix, Training vs Validation Loss).

The project consists of a **FastAPI backend** (hosted on Render) for handling predictions and model retraining, and a **React frontend** (deployed on Vercel) for a user-friendly interface. The backend leverages TensorFlow for machine learning, scikit-learn for metrics, and Matplotlib/Seaborn for visualizations. The frontend and backend are maintained in separate repositories.

## Links
- **FitVerse UI**: [https://fitverse-ui.vercel.app/](https://fitverse-ui.vercel.app/)
- **Video Demo**: [A video Demo - YouTube Link] *(Insert your YouTube video demo link here)*
- **Backend GitHub Repository**: [FitVerse GitHub Repo](https://github.com/g-tumwesigye/FitVerse)
- **Frontend GitHub Repository**: [FitVerse_UI GitHub Repo](https://github.com/g-tumwesigye/Fitverse_UI)

## Dataset
The dataset used was sourced from Kaggle:
- **Dataset Source**: [Fitness Exercises using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)

## Features
- **BMI Case Prediction**: Predict your BMI case by entering weight, height, age, BMI, and gender.
- **Model Retraining**: Upload a new dataset (CSV) to retrain the model and view performance metrics and visualizations.
- **Responsive UI**: A clean and intuitive interface built with React, featuring modals for prediction and retraining.
- **Cross-Origin Support**: The backend includes CORS middleware for seamless frontend-backend communication.

## Tech Stack
- **Frontend**: React, JavaScript, CSS
- **Backend**: FastAPI, Python, TensorFlow, scikit-learn, Matplotlib, Seaborn
- **Deployment**:
  - Frontend: Vercel
  - Backend: Render
- **API Endpoints**:
  - `/predict`: For BMI case prediction
  - `/retrain`: For retraining the model with a new dataset

## Project Structure
This repository contains the backend and related files. The frontend is maintained in a separate repository.

```
FitVerse/
├── README.md              
├── data/                  
│   ├── test/              
│   │   ├── X_test.csv    
│   │   └── y_test.csv     
│   └── train/             
│       ├── X_train.csv    
│       └── y_train.csv    
├── main.py                
├── models/                
│   ├── label_encoders.pkl 
│   ├── scaler.pkl         
│   └── sgd_momentum_model.h5 
├── notebook/             
│   └── FitVerse.ipynb     
├── requirements.txt       
└── src/                   
    ├── model.py           
    ├── prediction.py      
    ├── preprocessing.py   
    └── retrain.py         
```

## Setup Instructions

### Prerequisites
- **Python 3.8+** (for backend)
- **Node.js 14+** (for frontend)
- **Git**

### Step 1: Clone the Backend Repository
Clone the FitVerse backend repository from GitHub:
```bash
git clone https://github.com/g-tumwesigye/FitVerse.git
cd FitVerse
```

### Step 2: Set Up the Backend
1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Backend Locally**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The backend will be available at `http://localhost:8000`.

### Step 3: Set Up the Frontend
1. **Clone the Frontend Repository**:
   ```bash
   git clone https://github.com/g-tumwesigye/Fitverse_UI.git
   cd Fitverse_UI
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Run the Frontend Locally**:
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`.

### Step 4: Test Locally
- **Backend**: Test the backend endpoints using a tool like Postman:
  - POST `http://localhost:8000/predict` with a JSON body like:
    ```json
    {"Weight": 78, "Height": 1.76, "BMI": 23, "Age": 32, "Gender": "Male"}
    ```
  - POST `http://localhost:8000/retrain` by uploading a `.csv` file.
- **Frontend**: If running locally, open `http://localhost:3000` in your browser, click "Predict" to test the BMI prediction modal, and "Retrain" to test the retraining feature with a `.csv` file.

## Deployment

### Backend Deployment (Render)
1. **Deploy on Render**:
   - Log in to Render (https://dashboard.render.com/).
   - Select your backend service (`fitverse-q8be`).
   - Ensure the build command is `pip install -r requirements.txt` and the start command is `uvicorn main:app --host 0.0.0.0 --port $PORT`.
   - Deploy the service.
   - Backend URL: `https://fitverse-q8be.onrender.com/`

### Frontend Deployment (Vercel)
The frontend is in a separate repository. If you have access:
1. **Deploy on Vercel**:
   - Log in to Vercel (https://vercel.com/).
   - Select your frontend project.
   - Deploy the project.
   - Frontend URL: `https://fitverse-ui.vercel.app/`

## Testing
1. **Predict Modal**:
   - Open the frontend (`https://fitverse-ui.vercel.app/`).
   - Click "Predict", enter values (e.g., Weight: 78, Height: 1.76, Age: 32, BMI: 23, Gender: Male), and click "Predict BMI Case".
   - Verify the result (e.g., "Your predicted BMI case is: normal").

2. **Retrain Model**:
   - Click "Retrain", upload a `.csv` file with columns `Weight`, `Height`, `BMI`, `Age`, `BMIcase`, `Gender`.
   - Click "Retrain Model".
   - Verify the result shows the message, metrics (Test Loss, Accuracy, Precision, Recall, F1 Score, ROC AUC), and visualizations (Confusion Matrix, Loss Plot).

## Author
Geofrey Tumwesigye
