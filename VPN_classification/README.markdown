# ISCXVPN2016 VPN Traffic Classification Pipeline

## Overview

This project implements a machine learning pipeline for classifying network traffic as VPN or Non-VPN using the ISCXVPN2016 dataset. Designed to run in a Python environment (e.g., Jupyter Notebook or Google Colab), the pipeline includes data preprocessing, model training, testing, and real-time evaluation. It processes time-based traffic features, trains a Random Forest model for binary classification, and supports anomaly detection for potential AI-enhanced traffic. The pipeline is structured across Python scripts or notebooks, with saved artifacts for deployment and testing.

## Purpose

The pipeline aims to:

- Preprocess the ISCXVPN2016 dataset to extract time-based features and prepare it for classification.
- Train a Random Forest model to distinguish VPN from Non-VPN traffic with high accuracy.
- Enable real-time classification of network flows for network security applications.
- Detect anomalies that may indicate AI-enhanced or scripted traffic (e.g., low `std_flowiat`).
- Provide performance metrics, feature importance analysis (via SHAP), and visualizations.
- Support integration with datasets like UNSW-NB15, CICIDS2017, and CTU-13 for broader threat detection.

## Project Structure

The pipeline is organized into key components, typically implemented in Python scripts or Jupyter notebooks:

### 1. Data Preprocessing

- **Purpose**: Loads, cleans, and transforms the ISCXVPN2016 dataset to prepare it for model training.
- **Steps**:
  1. **Libraries and Setup**: Imports `pandas`, `numpy`, `sklearn`, etc.
  2. **Data Loading**: Loads `Combined_TimeBasedFeatures_Dataset.csv` (or raw ISCXVPN2016 data).
  3. **Data Cleaning**: Handles missing values (filled with median) and outliers.
  4. **Feature Selection**: Selects top 5 features via SHAP analysis: `min_flowiat`, `max_flowiat`, `std_flowiat`, `flowBytesPerSecond`, `max_fiat`.
  5. **Feature Engineering**: Scales features using `StandardScaler` and encodes `class1` (VPN/Non-VPN) with `LabelEncoder`.
  6. **Real-Time Helper**: Defines a preprocessing function for single-flow inputs.
- **Outputs**:
  - Scaler: `scaler.pkl`
  - Label Encoder: `label_encoder.pkl`
  - Feature names: `top_features.txt`
  - Processed dataset: Used for training and testing.
- **Key Metrics**: Processes thousands of flows, selects 5 features, handles slightly imbalanced classes (VPN: ~52.5%, Non-VPN: ~47.5%).

### 2. Model Training

- **Purpose**: Trains a Random Forest model for VPN vs. Non-VPN classification.
- **Steps**:
  1. **Libraries and Setup**: Imports `sklearn`, `pandas`, `numpy`, etc.
  2. **Data Loading**: Loads preprocessed data with top 5 features.
  3. **Model Training**: Trains `RandomForestClassifier` (50 estimators, balanced class weights) with 5-fold cross-validation.
  4. **Feature Importance**: Uses SHAP to identify top features (e.g., `min_flowiat` most impactful).
  5. **Performance Metrics**: Evaluates accuracy, precision, recall, and F1-score.
  6. **Visualization**: Generates confusion matrices and SHAP plots.
  7. **Model Saving**: Saves trained model and artifacts.
- **Outputs**:
  - Model: `vpn_classifier_model.pkl`
  - Metrics: Printed or saved (e.g., `dataset_metrics.txt`)
  - Visualizations: Confusion matrix (`dataset_confusion_matrix.png`), SHAP plots.
- **Key Metrics**:
  - Test Accuracy: 0.9644
  - Test Precision: 0.9520
  - Test Recall: 0.9818
  - Test F1-Score: 0.9667
  - Cross-Validation F1-Score: 0.9618 (±0.0017)

### 3. Testing and Anomaly Detection

- **Purpose**: Tests the model on `Combined_TimeBasedFeatures_Dataset.csv` and detects anomalies for AI-enhanced traffic.
- **Steps**:
  1. **Libraries and Setup**: Imports `sklearn`, `pandas`, `matplotlib`, `seaborn`, etc.
  2. **Load Artifacts**: Loads `vpn_classifier_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`.
  3. **Data Loading**: Loads `Combined_TimeBasedFeatures_Dataset.csv`.
  4. **Preprocessing**: Scales features and encodes labels.
  5. **Model Testing**: Evaluates performance on test data.
  6. **Anomaly Detection**: Uses `IsolationForest` to identify outliers (e.g., low `std_flowiat`).
  7. **Real-Time Simulation**: Tests predictions on individual flows with latency measurement.
  8. **Visualization**: Plots confusion matrices and anomaly scatter plots.
- **Outputs**:
  - Metrics: `dataset_metrics.txt`
  - Predictions: `dataset_with_predictions.csv` (if no labels)
  - Visualizations: `dataset_confusion_matrix.png`, `dataset_anomaly_scatter.png`
  - Anomaly model: `anomaly_detector.pkl`
- **Key Metrics**:
  - Anomaly detection: ~5% of flows flagged as outliers.
  - Real-time latency: <1ms per prediction.

### 4. Real-Time Evaluation (Optional)

- **Purpose**: Simulates real-time classification of network flows.
- **Steps**:
  1. **Load Artifacts**: Loads model, scaler, label encoder, and feature names.
  2. **Simulate Flows**: Uses test data or synthetic flows.
  3. **Preprocess Flows**: Applies scaling and feature selection.
  4. **Predict**: Classifies flows as VPN/Non-VPN and flags anomalies.
  5. **Evaluate**: Measures latency and accuracy.
- **Outputs**:
  - Predictions: Saved or printed.
  - Visualizations: Latency plots, prediction distributions.
- **Key Metrics**: Expected latency <1ms, accuracy/F1 ~0.9667.

## Prerequisites

- **Environment**: Python 3.8+ (Jupyter Notebook, Google Colab, or local setup).
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `flask` (for deployment).
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn joblib flask
  ```
- **Dataset**: `Combined_TimeBasedFeatures_Dataset.csv` with features like `min_flowiat`, `max_flowiat`, `std_flowiat`, `flowBytesPerSecond`, `max_fiat`, and `class1`.
- **Google Drive** (if using Colab):
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

## Setup Instructions

1. **Prepare Dataset**:
   - Place `Combined_TimeBasedFeatures_Dataset.csv` in the working directory or `/content/drive/MyDrive/ISCXVPN2016/`.
2. **Install Dependencies**:
   - Run `pip install` command above or install via notebook cell.
3. **Place Artifacts**:
   - Ensure `vpn_classifier_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`, and optionally `anomaly_detector.pkl` are in the working directory.
4. **Run Pipeline**:
   - Execute preprocessing, training, and testing scripts sequentially.
   - For testing, use the provided script below.
5. **Verify Outputs**:
   - Check for `dataset_metrics.txt`, `dataset_confusion_matrix.png`, `dataset_anomaly_scatter.png`.

## Usage

### 1. Data Preprocessing

- Load and preprocess `Combined_TimeBasedFeatures_Dataset.csv`:
  ```python
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  data = pd.read_csv('Combined_TimeBasedFeatures_Dataset.csv')
  features = ['min_flowiat', 'max_flowiat', 'std_flowiat', 'flowBytesPerSecond', 'max_fiat']
  X = data[features].fillna(data[features].median())
  scaler = StandardScaler().fit(X)
  X_scaled = scaler.transform(X)
  with open('scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)
  ```

### 2. Model Training

- Train the Random Forest model (if not already saved):
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score
  model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
  model.fit(X_scaled, data['class1'])
  scores = cross_val_score(model, X_scaled, data['class1'], cv=5, scoring='f1')
  print(f"CV F1-Score: {scores.mean():.4f} (±{scores.std():.4f})")
  with open('vpn_classifier_model.pkl', 'wb') as f:
      pickle.dump(model, f)
  ```

### 3. Testing and Anomaly Detection

- Test the model on `Combined_TimeBasedFeatures_Dataset.csv`:
  ```python
  import pickle
  import pandas as pd
  import numpy as np
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt
  import seaborn as sns

  # Load components
  with open('vpn_classifier_model.pkl', 'rb') as file:
      model = pickle.load(file)
  with open('scaler.pkl', 'rb') as file:
      scaler = pickle.load(file)
  with open('label_encoder.pkl', 'rb') as file:
      le = pickle.load(file)
  with open('top_features.txt', 'r') as file:
      features = file.read().splitlines()

  # Load dataset
  data = pd.read_csv('Combined_TimeBasedFeatures_Dataset.csv')
  X = data[features].fillna(data[features].median())
  X_scaled = scaler.transform(X)
  y = le.transform(data['class1']) if 'class1' in data.columns else None

  # Evaluate
  if y is not None:
      y_pred = model.predict(X_scaled)
      print("Test Set Metrics:")
      print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
      print(f"Precision: {precision_score(y, y_pred):.4f}")
      print(f"Recall: {recall_score(y, y_pred):.4f}")
      print(f"F1-Score: {f1_score(y, y_pred):.4f}")
      cm = confusion_matrix(y, y_pred)
      ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot()
      plt.savefig('confusion_matrix.png')

  # Anomaly detection
  try:
      with open('anomaly_detector.pkl', 'rb') as file:
          anomaly_model = pickle.load(file)
      anomalies = anomaly_model.predict(X_scaled)
      print(f"Anomalies detected: {(anomalies == -1).sum()}")
      X['is_anomaly'] = anomalies
      sns.scatterplot(data=X, x='min_flowiat', y='std_flowiat', hue='is_anomaly', palette={1: 'blue', -1: 'red'})
      plt.savefig('anomaly_scatter.png')
  except FileNotFoundError:
      from sklearn.ensemble import IsolationForest
      anomaly_model = IsolationForest(contamination=0.05, random_state=42)
      anomaly_model.fit(X_scaled)
      with open('anomaly_detector.pkl', 'wb') as file:
          pickle.dump(anomaly_model, file)
  ```

### 4. Real-Time Evaluation

- Simulate real-time predictions:
  ```python
  def real_time_prediction(packet, model, scaler, features):
      packet_df = pd.DataFrame([packet], columns=features)
      packet_scaled = scaler.transform(packet_df)
      pred = model.predict(packet_scaled)[0]
      return le.inverse_transform([pred])[0]

  # Example packet
  packet = [500, 4000, 100, 800, 1200]  # Replace with real data
  print(f"Prediction: {real_time_prediction(packet, model, scaler, features)}")
  ```

### 5. Deployment

- Deploy as a Flask API:
  ```python
  from flask import Flask, request, jsonify
  import pickle
  import pandas as pd

  app = Flask(__name__)
  with open('vpn_classifier_model.pkl', 'rb') as file:
      model = pickle.load(file)
  with open('scaler.pkl', 'rb') as file:
      scaler = pickle.load(file)
  with open('label_encoder.pkl', 'rb') as file:
      le = pickle.load(file)
  with open('top_features.txt', 'r') as file:
      features = file.read().splitlines()

  @app.route('/predict', methods=['POST'])
  def predict():
      data = request.json['features']
      packet_df = pd.DataFrame([data], columns=features)
      packet_scaled = scaler.transform(packet_df)
      pred = model.predict(packet_scaled)[0]
      label = le.inverse_transform([pred])[0]
      return jsonify({'prediction': label})

  app.run(debug=True)
  ```
  Test with:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"features": [500, 4000, 100, 800, 1200]}' http://localhost:5000/predict
  ```

## Notes

- **Dataset**: `Combined_TimeBasedFeatures_Dataset.csv` contains time-based features, with slight class imbalance (VPN: ~6586, Non-VPN: ~5957 in test set).
- **Performance**: Achieves F1-Score of 0.9667, with `min_flowiat` as the top feature (per SHAP).
- **Anomaly Detection**: Flags ~5% of flows as potential AI-enhanced traffic (e.g., low `std_flowiat`).
- **Environment**: Tested in Python 3.8+; Colab requires Drive mounting for file access.
- **Limitations**:
  - Relies on time-based features; missing features (e.g., `max_biat`) may require imputation.
  - AI-enhanced traffic detection limited by dataset’s lack of specific labels.
- **Next Steps**:
  - Integrate with UNSW-NB15, CICIDS2017, and CTU-13 for malicious/botnet detection.
  - Test on live network captures (e.g., Wireshark).
  - Explore LSTM models for temporal patterns.

## Example Outputs

- **Preprocessing**:
  - "Selected 5 features: ['min_flowiat', 'max_flowiat', 'std_flowiat', 'flowBytesPerSecond', 'max_fiat']"
  - Artifacts: `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`
- **Training**:
  - "Test Accuracy: 0.9644, Test F1: 0.9667"
  - "Top feature: min_flowiat (SHAP importance)"
- **Testing**:
  - "Anomalies detected: 500"
  - Visualizations: `confusion_matrix.png`, `anomaly_scatter.png`
- **Real-Time**:
  - "Prediction: VPN, Latency: 0.000123 sec"

## License

This project is for educational and research purposes, using the ISCXVPN2016 dataset. Ensure compliance with dataset usage policies and cite appropriately in academic work.