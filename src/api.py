import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from src.model import MODEL_PATH, retrain_model, create_tf_datasets, augment_dataset
app = FastAPI(title="Malaria Diagnosis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded model successfully.")
else:
    model = None
    print("No existing model found.")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Helper function: preprocess single image

def preprocess_image(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Predict single image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if model is None:
        return JSONResponse(status_code=400, content={"error": "Model not loaded"})

    img_array = preprocess_image(file_path)
    pred_prob = model.predict(img_array)[0][0]
    pred_class = "Parasitized" if pred_prob > 0.5 else "Uninfected"

    return {"filename": file.filename, "predicted_class": pred_class, "probability": float(pred_prob)}


# Batch prediction

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        img_array = preprocess_image(file_path)
        pred_prob = model.predict(img_array)[0][0]
        pred_class = "Parasitized" if pred_prob > 0.5 else "Uninfected"
        results.append({"filename": file.filename, "predicted_class": pred_class, "probability": float(pred_prob)})

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(UPLOAD_DIR, "batch_predictions.csv")
    df.to_csv(csv_path, index=False)

    return FileResponse(csv_path, media_type="text/csv", filename="batch_predictions.csv")


# Visualization endpoint

@app.get("/visualize_predictions")
def visualize_predictions():
    csv_path = os.path.join(UPLOAD_DIR, "batch_predictions.csv")
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "No batch predictions found"})

    df = pd.read_csv(csv_path)

    # Plot class distribution
    class_dist_path = os.path.join(UPLOAD_DIR, "class_distribution.png")
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="predicted_class", palette="Set2")
    plt.title("Prediction Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(class_dist_path)
    plt.close()

    # Plot probability distribution
    prob_dist_path = os.path.join(UPLOAD_DIR, "probability_distribution.png")
    plt.figure(figsize=(6,4))
    sns.histplot(df["probability"], bins=20, kde=True, color="skyblue")
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.savefig(prob_dist_path)
    plt.close()

    # Return image URLs
    return {
        "class_distribution_image": "/visualize_predictions/class_distribution.png",
        "probability_distribution_image": "/visualize_predictions/probability_distribution.png"
    }

@app.get("/visualize_predictions/{image_name}")
def serve_image(image_name: str):
    image_path = os.path.join(UPLOAD_DIR, image_name)
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})
    return FileResponse(image_path, media_type="image/png")


# Retrain model endpoint

@app.post("/retrain")
def retrain():
    try:
        train_ds, val_ds = create_tf_datasets()
        train_ds = augment_dataset(train_ds)
        global model
        model, history = retrain_model(train_ds, val_ds)
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# Status

@app.get("/status")
def status():
    if model is None:
        return {"status": "Model not loaded"}
    else:
        return {"status": "Model loaded", "model_path": MODEL_PATH}