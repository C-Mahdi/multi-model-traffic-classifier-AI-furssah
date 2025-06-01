# Multi-Model Network Traffic Classification Pipeline

## Overview

This project implements a machine learning pipeline for network traffic classification using the ISCXVPN2016 and CICIDS2017 datasets, with a Streamlit-based interface for model evaluation. The pipeline preprocesses network traffic data, trains Random Forest models for multiple classification tasks, and provides a user-friendly interface for batch evaluation, real-time metrics, and visualizations. It supports two primary tasks: classifying VPN vs. Non-VPN traffic (ISCXVPN2016) and classifying attack types, traffic sources, and operator types (CICIDS2017). The pipeline is designed to run in a Python environment, with the Streamlit app (interface.py) serving as the evaluation frontend.

## Purpose

The pipeline aims to:

- Preprocess ISCXVPN2016 and CICIDS2017 datasets to extract relevant features for classification.
- Train Random Forest models for:
  - VPN vs. Non-VPN classification (ISCXVPN2016).
  - Attack type (13 classes), traffic source (2 classes), and operator type (3 classes) classification (CICIDS2017).
- Provide a Streamlit interface for evaluating saved models with real-time metrics, confusion matrices, and downloadable reports.
- Enable deployment-ready artifacts for network security applications.

## Datasets

### 1. ISCXVPN2016

 link:https://www.unb.ca/cic/datasets/vpn.html

- **Description**: A dataset capturing network traffic for VPN and Non-VPN activities, designed for evaluating encrypted traffic classification.
- **Source**: University of New Brunswick, Canada.
- **Records**: Thousands of network flows (test set: `Combined_TimeBasedFeatures_Dataset.csv`).
- **Features**: 16 time-based features, including:
  - `min_flowiat`: Minimum inter-arrival time of flow packets.
  - `max_flowiat`: Maximum inter-arrival time of flow packets.
  - `std_flowiat`: Standard deviation of inter-arrival times.
  - `flowBytesPerSecond`: Bytes per second in the flow.
  - `max_fiat`: Maximum forward inter-arrival time.
  - Others: `mean_fiat`, `max_biat`, `duration`, `flowPktsPerSecond`, etc.
- **Top Features**: Selected via SHAP analysis: `min_flowiat`, `max_flowiat`, `std_flowiat`, `flowBytesPerSecond`, `max_fiat`.
- **Target**: `class1` (binary: VPN=1, Non-VPN=0).
- **Class Distribution**: Slightly imbalanced (VPN: \~52.5%, Non-VPN: \~47.5% in test set).
- **Usage**: Used for VPN vs. Non-VPN classification and anomaly detection (e.g., AI-enhanced traffic).

### 2. CICIDS2017

link:https://www.unb.ca/cic/datasets/ids-2017.html

- **Description**: A dataset capturing network traffic with benign and malicious activities, designed for intrusion detection and threat classification.
- **Source**: University of New Brunswick, Canada.
- **Records**: 2,830,743 flows across 8 CSV files.
- **Features**: 80 features, including:
  - `Bwd Packet Length Max`: Maximum length of backward packets.
  - `Total Length of Fwd Packets`: Total length of forward packets.
  - `Fwd Packets/s`: Forward packets per second.
  - Others: Flow duration, packet counts, inter-arrival times, etc.
- **Top Features**: Selected via feature importance: `Bwd Packet Length Max`, `Total Length of Fwd Packets`, `Fwd Packets/s`.
- **Targets**:
  - Attack Type: 13 classes (e.g., DoS, Brute Force, Bot).
  - Traffic Source: 2 classes (e.g., benign, malicious).
  - Operator Type: 3 classes (e.g., different operator categories).
- **Class Distribution**: Imbalanced, with benign traffic dominating (\~80%).
- **Usage**: Preprocessed to 100,000 samples with 7 features for multi-classification tasks.


The pipeline is organized into preprocessing, training, and evaluation components, with the Streamlit interface as the evaluation frontend.

### 1. Data Preprocessing

- **Purpose**: Cleans and transforms ISCXVPN2016 and CICIDS2017 datasets for model training.
- **Components**:
  1. **Libraries and Setup**: Uses `pandas`, `numpy`, `sklearn`,`shap`, and related libraries.
  2. **Data Loading**:
     - ISCXVPN2016: Loads `Combined_TimeBasedFeatures_Dataset.csv`.
     - CICIDS2017: Combines 8 CSV files from `MachineLearningCVE/`.
  3. **Data Cleaning**:
     - ISCXVPN2016: Fills missing values with median.
     - CICIDS2017: Handles missing (e.g., 1358 in `Flow Bytes/s`) and infinite values.
  4. **Feature Engineering**:
     - ISCXVPN2016: Scales features using `StandardScaler`.
     - CICIDS2017: Creates protocol categories and flow metrics.
  5. **Feature Selection**:
     - ISCXVPN2016: Selects 5 SHAP-identified features.
     - CICIDS2017: Selects 7 features via importance analysis.
  6. **Target Preparation**:
     - ISCXVPN2016: Encodes `class1` with `LabelEncoder`.
     - CICIDS2017: Prepares 3 target variables (attack type, traffic source, operator type).
- **Outputs**:
  - ISCXVPN2016: `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`.
  - CICIDS2017: `scaler.joblib`, `feature_names.txt`, train-test splits.

### 2. Model Training

- **Purpose**: Trains Random Forest models for classification tasks.
- **Components**:
  1. **Libraries and Setup**: Uses `sklearn`, `pandas`, `numpy`, and visualization libraries.
  2. **Data Loading**: Loads preprocessed data.
  3. **Model Training**:
     - ISCXVPN2016: Trains `RandomForestClassifier` (50 estimators, balanced weights) with 5-fold CV for VPN classification.
     - CICIDS2017: Trains models for attack type, traffic source, and operator type with GridSearchCV (24 parameter combinations, 3-fold CV).
  4. **Feature Importance**:
     - ISCXVPN2016: SHAP analysis identifies `min_flowiat` as most impactful.
     - CICIDS2017: Identifies `Bwd Packet Length Max` as key feature.
  5. **Results Compilation**: Saves metrics and visualizations.
- **Outputs**:
  - ISCXVPN2016: `vpn_classifier_model.pkl`.
  - CICIDS2017: `attack_type_random_forest_model.joblib`, `traffic_source_random_forest_model.joblib`, `operator_type_random_forest_model.joblib`.
- **Key Metrics**:
  - ISCXVPN2016: Test Accuracy: 0.9644, F1-Score: 0.9667, CV F1: 0.9618 (±0.0017).
  - CICIDS2017:
    - Attack Type: Test Accuracy: 0.9801, F1: 0.9798, CV F1: 0.9801 (±0.0004).
    - Traffic Source: Test Accuracy: 0.9805, F1: 0.9806, CV F1: 0.9815 (±0.0007).
    - Operator Type: Test Accuracy: 0.9809, F1: 0.9806, CV F1: 0.9812 (±0.0012).

### 3. Streamlit Interface (interface.py)

- **Purpose**: Provides a web-based interface for evaluating saved models on test data.
- **Components**:
  1. **Model Selection**: Allows selection from four models (VPN Classifier, Attack Type, Traffic Source, Operator Type).
  2. **Data Upload**:
     - ISCXVPN2016: Single CSV (`Combined_TimeBasedFeatures_Dataset.csv`) with features and labels.
     - CICIDS2017: Separate X (`X_test.csv`) and Y (`y_test.csv`) files for each task.
  3. **Data Preview**: Displays feature statistics, label distribution, and data shape.
  4. **Evaluation Parameters**: Configures batch size (default: 1000) and warmup runs (default: 2).
  5. **Batch Evaluation**:
     - Processes data in batches for efficiency.
     - Computes accuracy, precision, recall, F1-score, and timing metrics (time/sample, throughput).
     - Updates real-time metrics and charts (accuracy, timing, class distribution).
  6. **Results Visualization**:
     - Global confusion matrix with class names.
     - Real-time performance plots (accuracy, precision, recall, F1-score).
     - Classification report with per-class metrics.
  7. **Downloadable Outputs**:
     - Batch results CSV (per-batch metrics).
     - Global summary CSV (overall performance).
- **Outputs**:
  - Visualizations: Real-time charts, confusion matrix.
  - Reports: CSV files for batch and global results.
- **Key Metrics**: 
- accuracy\\F1 consistent with test set (\~0.9667 for ISCXVPN2016, \~0.98 for CICIDS2017).

## Prerequisites

- **Environment**: Python 3.8+ (local or Google Colab).
- **Python Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `pickle`, `streamlit`, `plotly`.
- **Datasets**:
  - ISCXVPN2016: `Combined_TimeBasedFeatures_Dataset.csv`.
  - CICIDS2017: Raw CSV files in `MachineLearningCVE/` or preprocessed test files (`attack_type_X_test.csv`, etc.).
- **Model Artifacts**:
  - ISCXVPN2016: `vpn_classifier_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `top_features.txt`.
  - CICIDS2017: `attack_type_random_forest_model.joblib`, `traffic_source_random_forest_model.joblib`, `operator_type_random_forest_model.joblib`.
- **Google Drive** (if using Colab): Mount Drive to access datasets and artifacts.

## Setup Instructions

1. **Prepare Datasets**:
   - ISCXVPN2016: Place `Combined_TimeBasedFeatures_Dataset.csv` in the working directory or `/content/drive/MyDrive/ISCXVPN2016/`.
   - CICIDS2017: Place raw CSVs in `/content/drive/MyDrive/MachineLearningCVE/` or test files (`X_test.csv`, `y_test.csv`) in the working directory.
2. **Install Dependencies**:
   - Install libraries via `pip install pandas numpy scikit-learn matplotlib seaborn joblib pickle streamlit plotly`.
3. **Place Artifacts**:
   - Ensure model files, scaler, and feature lists are in the working directory.
4. **Run Streamlit App**:
   - Execute `streamlit run interface.py` to launch the interface.
5. **Verify Outputs**:
   - Check for CSVs and visualizations generated by the Streamlit app.

## Usage

1. **Data Preprocessing**:

   - ISCXVPN2016: Process `Combined_TimeBasedFeatures_Dataset.csv` to select 5 features, handle missing values, and scale data.
   - CICIDS2017: Combine 8 CSVs, select 7 features, and sample 100,000 records.

2. **Model Training**:

   - ISCXVPN2016: Train Random Forest model with 5-fold CV for VPN classification.
   - CICIDS2017: Train models for three tasks with GridSearchCV.

3. **Streamlit Interface**:

   - Launch `interface.py` to access the web interface.
   - Select a model (e.g., VPN Classifier).
   - Upload data:
     - ISCXVPN2016: Single CSV with features and labels.
     - CICIDS2017: Separate X and Y CSVs.
   - Configure batch size and warmup runs.
   - Run batch evaluation to view real-time metrics and visualizations.
   - Download batch and global result CSVs.

## Notes

- **Datasets**:
  - ISCXVPN2016: Time-based features, slight imbalance (VPN: \~52.5%).
  - CICIDS2017: 2,830,743 records, sampled to 100,000, imbalanced (benign: \~80%).
- **Performance**:
  - ISCXVPN2016: F1-Score: 0.9667, top feature: `min_flowiat`.
  - CICIDS2017: F1-Scores: \~0.98, top feature: `Bwd Packet Length Max`.
- **Streamlit Interface**: Supports numeric/string labels, real-time visualization, and downloadable reports.
- **Limitations**:
  - ISCXVPN2016: Missing features (e.g., `max_biat`) may require imputation.
  - CICIDS2017: Imbalanced classes may affect minority class performance.
  - AI-enhanced traffic detection limited by dataset labels.

## Example Outputs

- **Preprocessing**:
  - ISCXVPN2016: "Selected 5 features: \['min_flowiat', ...\]"
  - CICIDS2017: "Processed 2,830,743 records, sampled 100,000"
- **Training**:
  - ISCXVPN2016: "Test F1: 0.9667"
  - CICIDS2017: "Attack Type F1: 0.9798"
- **Streamlit Interface**:
  - "Global Accuracy: 0.9644 for VPN Classifier"
  - Visualizations: Confusion matrices, real-time performance plots.
  - CSVs: Batch results, global summary.
- **Anomaly Detection**:
  - "Anomalies detected: \~500 flows"

## 🤝 Collaborators

 
- [@MahdiChaabani](https://github.com/C-Mahdi)
- [@MoatazBenTrad](https://github.com/trad024)
- [@FadouaMili](https://github.com/fadoua1m)
- [@AhmedLakti](https://github.com/Ahmed-Lakti)



## License

This project is for educational and research purposes, using ISCXVPN2016 and CICIDS2017 datasets. Ensure compliance with dataset usage policies and cite appropriately in academic wo
