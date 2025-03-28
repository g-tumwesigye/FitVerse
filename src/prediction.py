import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Loading saved model, scaler & label_encoders
loaded_model = load_model("/content/saved_models/sgd_momentum_model.h5")
scaler = joblib.load("/content/saved_models/scaler.pkl")
label_encoders = joblib.load("/content/saved_models/label_encoders.pkl")

# test examples and then predict
sample_input = X_test[:5]
predictions = loaded_model.predict(sample_input)

# Converting predicted class indexes back to original labels
predicted_classes = np.argmax(predictions, axis=1)
decoded_classes = label_encoders["BMIcase"].inverse_transform(predicted_classes)

# Display results
for i, pred in enumerate(decoded_classes):
    print(f"Sample {i+1}: BMICase -> {pred}")
