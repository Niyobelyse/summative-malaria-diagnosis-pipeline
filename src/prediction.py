import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# Load the saved model

model_path = "/home/belysetag/summative-malaria-diagnosis-pipeline/models/malaria_diagnosis"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")


# Preprocess a single image

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array


# Predict class for a single image

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    pred_prob = model.predict(img_array)[0][0]
    pred_class = "Parasitized" if pred_prob > 0.5 else "Uninfected"
    return pred_class, float(pred_prob)


# Batch prediction

def predict_batch(image_dir, save_csv=True):
    results = []
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            pred_class, pred_prob = predict_image(img_path)
            results.append({
                "filename": img_file,
                "predicted_class": pred_class,
                "probability": pred_prob
            })
            print(f"Predicted {img_file}: {pred_class} ({pred_prob:.3f})")

    if save_csv:
        df = pd.DataFrame(results)
        save_dir = "/home/belysetag/summative-malaria-diagnosis-pipeline/models"
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "batch_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Batch predictions saved at {csv_path}")

    return results


# Example usage

if __name__ == "__main__":
    image_dir = "/home/belysetag/summative-malaria-diagnosis-pipeline/new_images"
    predict_batch(image_dir)