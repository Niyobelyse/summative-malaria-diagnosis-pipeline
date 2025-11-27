# Malaria Diagnosis Pradiction

## Overview

This project is  MLOps pipeline for malaria diagnosis using deep learning. It includes data acquisition, preprocessing, model training, evaluation, retraining, API deployment, and a user-friendly web UI. The solution is designed for cloud deployment and supports real-time prediction and retraining triggers.



## Features

- **Data Acquisition & Processing:** Automated loading and preprocessing of malaria cell images.
- **Model Creation & Training:** Transfer learning with VGG16, fine-tuning, and regularization.
- **Model Evaluation:** Metrics include accuracy, loss, F1 score, precision, recall, and AUC-ROC.
- **Prediction:** Single and batch image prediction via FastAPI and Streamlit UI.
- **Retraining:** Upload new data and trigger retraining from the UI or API.
- **Visualizations:** Interactive charts for predictions and model performance.
- **Deployment:** Docker-ready, cloud deployable.
- **MLOps:** Automated retraining, monitoring, and evaluation.
- **Load Testing:** Locust scripts for API stress testing.


## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Train and save the model:**
   ```sh
   python src/model.py
   ```

3. **Start FastAPI backend:**
   ```sh
   uvicorn src.api:app --reload
   ```

4. **Start Streamlit UI:**
   ```sh
   streamlit run src/app.py
   ```

5. **Access the UI:**
   - [http://localhost:8501](http://localhost:8501)

---

## API Endpoints
![alt text](<Screenshot from 2025-11-27 19-52-18.png>)
- `/predict` - Predict single image
- `/predict_batch` - Predict batch images
- `/retrain` - Trigger model retraining
- `/visualize_predictions` - Get prediction visualizations
- `/status` - Model status

---

## Evaluation

- **Preprocessing:** Automated, with augmentation.
- **Optimization:** Transfer learning, fine-tuning, regularization, Adam optimizer.
- **Metrics:** Accuracy, loss, F1 score, precision, recall, AUC-ROC.
- **Notebook:** See `notebooks/model_evaluation.ipynb` for details.

---

## Deployment

- **Docker:** Use the provided Dockerfile to containerize the app.
- **Cloud:** Deploy on AWS, Azure, GCP, or Heroku.
- **Load Testing:** Use Locust scripts to simulate API requests and measure latency.

---


## How to Retrain

- Upload new training data via UI or API.
- Click "Start Retraining" in the UI or call `/retrain` endpoint.
- Model will be retrained and saved automatically.

---

## How to Predict

- Upload an image via UI or API.
- Click "Predict" to get diagnosis and probability.

---

## Load Testing

- Use Locust to simulate requests:
  ```sh
  locust -f locustfile.py
  ```
- Monitor latency and response time with different container counts.

---

## Video Demo

- See attached video for a demonstration of prediction and retraining with camera on.

https://www.youtube.com/watch?v=2pSWr5rNT00

---

## Notes

- Ensure `models/malaria_diagnosis.h5` exists before using prediction endpoints.
- For best results, deploy with Docker and monitor using provided tools.

---
