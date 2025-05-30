# ISCXVPN2016 VPN Traffic Classification Pipeline

## Overview

This project implements a machine learning pipeline for classifying network traffic as VPN or Non-VPN using the ISCXVPN2016 dataset. Designed to run in a Python environment (e.g., Jupyter Notebook or Google Colab), the pipeline consists of three main components: data preprocessing, model training, and real-time evaluation. It processes time-based traffic features, trains a Random Forest model for binary classification, and supports anomaly detection for potential AI-enhanced traffic. The pipeline is structured across Python scripts or notebooks, with saved artifacts for deployment.

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
- **Components**:
  1. **Libraries and Setup**: Uses `pandas`, `numpy`, `sklearn`, and related libraries.
  2. **Configuration**: Defines file paths for input data and output artifacts.
  3. **Data Loading and Exploration**: Loads `Combined_TimeBasedFeatures_Dataset.csv` and analyzes feature distributions.
  4. **Data Cleaning**: Handles missing values (filled with median) and outliers.
  5. **Feature Engineering**: Scales features using `StandardScaler` and encodes `class1` (VPN/Non-VPN) with `LabelEncoder`.
  6. **Feature Selection**: Selects top 5 features via SHAP analysis: `min_flowiat`, `max_flowiat`, `std_flowiat`, `flowBytesPerSecond`, `max_fiat`.
  7. **Real-Time Helper**: Prepares functions for single-flow preprocessing.
- **Outputs**:
  - Scaler: `scaler.pkl`
  - Label Encoder: `label_encoder.pkl`
  - Feature names: `top_features.txt`
- **Key Metrics**: Processes thousands of flows, selects 5 features, handles slightly imbalanced classes (VPN: ~52.5%, Non-VPN: ~47.5%).

### 2. Model Training

- **Purpose**: Trains a Random Forest model for VPN vs. Non-VPN classification.
- **Components**:
  1. **Libraries and Setup**: Uses `sklearn`, `pandas`, `numpy`, and visualization libraries.
  2. **Configuration**: Specifies paths for preprocessed data and model outputs.
  3. **Data Loading**: Loads preprocessed data with top 5 features.
  4. **Model Training**: Trains `RandomForestClassifier` (50 estimators, balanced class weights) with 5-fold cross-validation.
  5. **Feature Importance Analysis**: Uses SHAP to identify top features (e.g., `min_flowiat` most impactful).
  6. **Results Compilation**: Generates performance metrics and visualizations.
- **Outputs**:
  - Model: `vpn_classifier_model.pkl`
  - Visualizations: SHAP plots and confusion matrices.
- **Key Metrics**:
  - Test Accuracy: 0.9644
  - Test Precision: 0.9520
  - Test Recall: 0.9818
  - Test F1-Score: 0.9667
  - Cross-Validation F1-Score: 0.9618 (±0.0017)

### 3. Real-Time Evaluation

- **Purpose**: Simulates real-time classification of network flows using trained models and preprocessing artifacts.
- **Components**:
  1. **Libraries and Setup**: Uses `sklearn`, `pandas`, and related libraries.
  2. **Configuration**: Defines paths to models, scaler, and feature names.
  3. **Load Artifacts**: Loads `vpn_classifier_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `top_features.txt`.
  4. **Simulate Network Flows**: Processes sample flows from test data or synthetic inputs.
  5. **Preprocess Flows**: Applies scaling and feature selection.
  6. **Real-Time Prediction**: Classifies flows as VPN/Non-VPN.
  7. **Results Visualization**: Generates prediction distributions and latency metrics.
- **Outputs**:
  - Prediction results: Saved as CSV files.
  - Visualizations: Latency and prediction plots.
- **Key Metrics**: Expected latency <1ms per prediction, accuracy/F1 ~0.9667.

### 4. Anomaly Detection

- **Purpose**: Identifies potential AI-enhanced or scripted traffic using `IsolationForest`.
- **Components**:
  1. **Model Training**: Fits `IsolationForest` on scaled features to detect outliers.
  2. **Analysis**: Flags flows with anomalies (e.g., low `std_flowiat`, consistent `mean_fiat`).
  3. **Visualization**: Plots scatter diagrams of anomalies (e.g., `min_flowiat` vs. `std_flowiat`).
- **Outputs**:
  - Anomaly model: `anomaly_detector.pkl`
  - Visualizations: Anomaly scatter plots.
- **Key Metrics**: Flags ~5% of flows as potential anomalies.

## Prerequisites

- **Environment**: Python 3.8+ (Jupyter Notebook, Google Colab, or local setup).
- **Python Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `flask` (for deployment).
- **Dataset**: `Combined_TimeBasedFeatures_Dataset.csv` with features like `min_flowiat`, `max_flowiat`, `std_flowiat`, `flowBytesPerSecond`, `max_fiat`, and `class1`.
- **Google Drive** (if using Colab): Mount Drive to access dataset and artifacts.

## Setup Instructions

1. **Prepare Dataset**:
   - Place `Combined_TimeBasedFeatures_Dataset.csv` in the working directory or `/content/drive/MyDrive/ISCXVPN2016/`.
2. **Install Dependencies**:
   - Install required libraries via `pip` or notebook cells.
3. **Place Artifacts**:
   - Ensure `vpn_classifier_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`, and optionally `anomaly_detector.pkl` are available.
4. **Run Pipeline**:
   - Execute preprocessing, training, and evaluation components sequentially.
5. **Verify Output Directories**:
   - Preprocessing: `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`
   - Training: `vpn_classifier_model.pkl`
   - Anomaly Detection: `anomaly_detector.pkl`

## Usage

1. **Data Preprocessing**:
   - Process `Combined_TimeBasedFeatures_Dataset.csv` to select top 5 features, handle missing values, and scale data.
   - Save preprocessing artifacts for training and evaluation.

2. **Model Training**:
   - Train Random Forest model on preprocessed data with 5-fold cross-validation.
   - Analyze feature importance via SHAP and save model.

3. **Real-Time Evaluation**:
   - Simulate real-time classification using saved model and preprocessing artifacts.
   - Process sample flows and measure prediction latency.

4. **Anomaly Detection**:
   - Apply `IsolationForest` to detect AI-enhanced traffic.
   - Visualize anomalies for further analysis.

5. **Folder Structure**:
   ```
   /ISCXVPN2016/
   ├── Combined_TimeBasedFeatures_Dataset.csv  # Dataset
   ├── vpn_classifier_model.pkl               # Trained model
   ├── scaler.pkl                             # Scaler
   ├── label_encoder.pkl                      # Label encoder
   ├── top_features.txt                       # Feature names
   ├── anomaly_detector.pkl                   # Anomaly model
   ├── preprocessing_output/                  # Preprocessing artifacts
   ├── training_output/                       # Training artifacts
   │   ├── visualizations/
   │   └── results/
   └── evaluation_output/                     # Evaluation artifacts
       ├── predictions/
       └── visualizations/
   ```

## Notes

- **Dataset**: `Combined_TimeBasedFeatures_Dataset.csv` contains time-based features, with slight class imbalance (VPN: ~52.5%, Non-VPN: ~47.5%).
- **Performance**: Achieves F1-Score of 0.9667, with `min_flowiat` as the top feature (per SHAP).
- **Anomaly Detection**: Flags ~5% of flows as potential AI-enhanced traffic (e.g., low `std_flowiat`).
- **Environment**: Compatible with Python 3.8+; Colab requires Drive mounting.
- **Limitations**:
  - Relies on time-based features; missing features (e.g., `max_biat`) may require imputation.
  - AI-enhanced traffic detection limited by dataset’s lack of specific labels.
- **Next Steps**:
  - Integrate with UNSW-NB15, CICIDS2017, and CTU-13 for malicious/botnet detection.
  - Validate on live network captures (e.g., Wireshark).
  - Explore LSTM models for temporal patterns.

## Example Outputs

- **Preprocessing**:
  - "Selected 5 features: ['min_flowiat', 'max_flowiat', 'std_flowiat', 'flowBytesPerSecond', 'max_fiat']"
  - Artifacts: `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`
- **Training**:
  - "Test Accuracy: 0.9644, Test F1: 0.9667"
  - "Top feature: min_flowiat (SHAP importance)"
- **Evaluation**:
  - "Processed flows with average latency: <1ms"
  - "Real-time accuracy: ~0.9644, F1: ~0.9667"
- **Anomaly Detection**:
  - "Anomalies detected: ~500 flows"
  - Visualizations: Scatter plots of anomalies.

## License

This project is for educational and research purposes, using the ISCXVPN2016 dataset. Ensure compliance with dataset usage policies and cite appropriately in academic work.