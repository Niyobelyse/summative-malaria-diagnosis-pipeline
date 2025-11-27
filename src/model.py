import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam


# CONFIG

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DROPOUT_RATE = 0.5
SAVE_DIR = "/home/belysetag/summative-malaria-diagnosis-pipeline/models"
MODEL_PATH = os.path.join(SAVE_DIR, "malaria_diagnosis")


# Load datasets functions

from src.preprocessing import create_tf_datasets, augment_dataset
train_ds, test_ds = create_tf_datasets()
train_ds = augment_dataset(train_ds)  # only augment the training set


# Build VGG16 Models

def build_vgg16_transfer(input_shape=(224,224,3), dropout_rate=DROPOUT_RATE):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_vgg16_finetune(input_shape=(224,224,3), dropout_rate=DROPOUT_RATE, fine_tune_at=4):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_at:]:
        layer.trainable = True
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Training & Evaluation Functions

def plot_learning_curves(history, title="Learning Curves"):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, dataset, class_names=['Parasitized','Uninfected']):
    y_prob = model.predict(dataset)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_true = np.concatenate([y.numpy() for x, y in dataset], axis=0)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# Retrain Function

def retrain_model(new_train_ds, new_val_ds, epochs=EPOCHS, fine_tune_at=4):
    if os.path.exists(MODEL_PATH):
        print("Loading existing model for retraining...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No existing model found. Creating a new model...")
        model = build_vgg16_finetune(fine_tune_at=fine_tune_at)

    history = model.fit(new_train_ds, validation_data=new_val_ds, epochs=epochs)

    # Save updated model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Retrained model saved at {MODEL_PATH}")

    # Save training history
    history_path = os.path.join(SAVE_DIR, "malaria_diagnosis_retrain_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print(f"Retraining history saved at {history_path}")

    return model, history


# Main Training

if __name__ == "__main__":
    # Train new or fine-tuned model
    model = build_vgg16_finetune(fine_tune_at=4)
    history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

    plot_learning_curves(history, "VGG16 Fine-Tuned")
    evaluate_model(model, test_ds)

    # Save trained model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Fine-tuned model saved at {MODEL_PATH}")

    # Save history
    history_path = os.path.join(SAVE_DIR, "malaria_diagnosis_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print(f"Training history saved at {history_path}")

    # Example: Save your trained Keras model
    model.save("/home/belysetag/summative-malaria-diagnosis-pipeline/models/malaria_diagnosis.h5")