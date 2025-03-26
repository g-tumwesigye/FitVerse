import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

def train_save_evaluate(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, save_dir="/content/saved_models", callbacks=None):
    os.makedirs(save_dir, exist_ok=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    model.save(os.path.join(save_dir, f"{model_name}.h5"))
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    print(f"\n{model_name} Results:")
    print(f" Test Loss: {test_loss:.4f}")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 Score: {f1:.4f}")
    return history, y_pred_probs, y_pred_classes, y_true

# Define EarlyStopping callback
early_stop_sgd = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Define the Neural Network Model
sgd_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],),
          kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

# Compile the Model
sgd_model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.95),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train, save, and evaluate
history_sgd, y_pred_probs, y_pred_classes, y_true = train_save_evaluate(
    sgd_model, "sgd_momentum_model",
    X_train_resampled, y_train_resampled,
    X_val, y_val, X_test, y_test,
    callbacks=[early_stop_sgd]
)
