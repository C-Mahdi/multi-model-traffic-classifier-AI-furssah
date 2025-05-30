# CICIDS2017 Network Traffic Classification Pipeline

## Overview

This project implements a complete machine learning pipeline for network traffic classification and threat detection using the CICIDS2017 dataset. Designed to run in Google Colab, the pipeline consists of three main components: data preprocessing, model training, and real-time evaluation. It processes network traffic data, trains Random Forest models for multiple classification tasks, and enables real-time prediction for network flows. The pipeline is structured across three Jupyter notebooks: `data_preprocessing.ipynb`, `models_training.ipynb`, and `realtime_evaluation.ipynb`.

## Purpose

The pipeline aims to:

- Preprocess the CICIDS2017 dataset to prepare it for machine learning, handling data cleaning, feature engineering, and target variable creation.
- Train Random Forest models for three classification tasks: attack type (13 classes), traffic source (2 classes), and operator type (3 classes).
- Enable real-time classification of network flows using trained models and preprocessing artifacts.
- Provide robust performance metrics, feature importance analysis, and visualizations for model evaluation.
- Support deployment-ready artifacts for real-time threat detection systems.

## Project Structure

The pipeline is organized into three Jupyter notebooks, each handling a distinct phase:

### 1. Data Preprocessing (`data_preprocessing.ipynb`)

- **Purpose**: Loads, cleans, and transforms the CICIDS2017 dataset to prepare it for model training.
- **Cells**:
  1. **Libraries and Setup**: Imports pandas, numpy, sklearn, etc.
  2. **Configuration**: Defines file paths and parameters (e.g., `DATA_PATH`, `OUTPUT_BASE`).
  3. **Data Loading and Exploration**: Combines 8 CSV files (2,830,743 records, 80 features) and prints statistics.
  4. **Data Quality Assessment**: Analyzes distributions (traffic type, source, operator).
  5. **Data Cleaning**: Handles missing (e.g., 1358 in `Flow Bytes/s`) and infinite values.
  6. **Feature Engineering**: Creates features like protocol categories and flow metrics.
  7. **Feature Selection and Preparation**: Selects 7 features, prepares 3 target variables, and samples 100,000 records.
  8. **Real-Time Helper Function**: Defines `preprocess_realtime_sample` for single-flow preprocessing.
- **Outputs**:
  - Scaler: `/content/drive/MyDrive/furssah/preprocessing_output/models/scaler.joblib`
  - Feature names: `/content/drive/MyDrive/furssah/preprocessing_output/features/feature_names.txt`
  - Train-test splits and encodings: `/content/drive/MyDrive/furssah/preprocessing_output/`
- **Key Metrics**: Processes 2,830,743 records, selects 7 features, samples 100,000 records.

### 2. Model Training (`models_training.ipynb`)

- **Purpose**: Trains Random Forest models for three classification tasks using preprocessed data.
- **Cells**:
  1. **Libraries and Setup**: Imports pandas, numpy, sklearn, matplotlib, etc.
  2. **Configuration**: Defines training paths and hyperparameter grid.
  3. **Data Loading**: Loads preprocessed data (80,000 train, 20,000 test samples per task). 4–6. **Model Training**: Trains models for attack type, traffic source, and operator type with GridSearchCV (24 parameter combinations, 3-fold CV).
  4. **Results Compilation**: Saves performance metrics to CSV.
  5. **Visualizations**: Generates confusion matrices and performance plots.
  6. **Feature Importance Analysis**: Identifies top features (e.g., `Bwd Packet Length Max`).
  7. **Final Summary**: Prints training summary and next steps.
- **Outputs**:
  - Models: `/content/drive/MyDrive/furssah/training_output/trained_models/{task}_random_forest_model.joblib`
  - Results: `/content/drive/MyDrive/furssah/training_output/results/random_forest_results.csv`
  - Visualizations: `/content/drive/MyDrive/furssah/training_output/visualizations/`
  - Feature importance: `/content/drive/MyDrive/furssah/training_output/feature_analysis/{task}_feature_importance.csv`
- **Key Metrics**:
  - **Attack Type**: Test Accuracy: 0.9801, Test F1: 0.9798, CV F1: 0.9801 (±0.0004)
  - **Traffic Source**: Test Accuracy: 0.9805, Test F1: 0.9806, CV F1: 0.9815 (±0.0007)
  - **Operator Type**: Test Accuracy: 0.9809, Test F1: 0.9806, CV F1: 0.9812 (±0.0012)
  - Total training time: \~1928.7 seconds (\~32.1 minutes).

### 3. Real-Time Evaluation (`realtime_evaluation.ipynb`)

- **Purpose**: Simulates real-time classification of network flows using trained models and preprocessing artifacts.
- **Cells (Hypothetical)**:
  1. **Libraries and Setup**: Imports necessary libraries (pandas, sklearn, joblib).
  2. **Configuration**: Defines paths to models, scaler, and feature names.
  3. **Load Artifacts**: Loads trained models, scaler, and feature names from preprocessing and training outputs.
  4. **Simulate Network Flows**: Generates or loads sample network flow data (e.g., from test set or synthetic data).
  5. **Preprocess Flows**: Applies `preprocess_realtime_sample` to prepare flows for prediction.
  6. **Real-Time Prediction**: Uses trained models to classify flows for all three tasks.
  7. **Performance Evaluation**: Computes latency, accuracy, and F1 score for real-time predictions.
  8. **Results Visualization**: Plots prediction distributions and latency metrics.
- **Outputs**:
  - Prediction results: `/content/drive/MyDrive/furssah/evaluation_output/realtime_predictions.csv`
  - Visualizations: `/content/drive/MyDrive/furssah/evaluation_output/visualizations/`
- **Key Metrics (Expected)**: Low-latency predictions (&lt;100ms per flow), accuracy/F1 consistent with test set (\~0.98).

## Prerequisites

- **Google Colab Environment**: All notebooks run in Google Colab with Google Drive access.
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib (installed in first cell of each notebook).
- **CICIDS2017 Dataset**:
  - Raw CSV files (8 files, 2,830,743 records) in `/content/drive/MyDrive/furssah/MachineLearningCVE/` for preprocessing.
  - Preprocessed artifacts in `/content/drive/MyDrive/furssah/preprocessing_output/` for training and evaluation.
- **Google Drive**: Mount Drive in Colab to access dataset and save outputs:

  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

## Setup Instructions

1. **Mount Google Drive**: Run the mount command in each notebook.
2. **Prepare Dataset**:
   - Place CICIDS2017 CSV files in `/content/drive/MyDrive/furssah/MachineLearningCVE/`.
   - Ensure preprocessed artifacts are available for training and evaluation (run `data_preprocessing.ipynb` first).
3. **Install Dependencies**: Execute the first cell of each notebook to import/install libraries.
4. **Run Notebooks Sequentially**:
   - **Preprocessing**: Run `data_preprocessing.ipynb` to generate preprocessed data.
   - **Training**: Run `models_training.ipynb` to train models.
   - **Evaluation**: Run `realtime_evaluation.ipynb` to simulate real-time predictions.
5. **Verify Output Directories**:
   - Preprocessing: `/content/drive/MyDrive/furssah/preprocessing_output/`
   - Training: `/content/drive/MyDrive/furssah/training_output/`
   - Evaluation: `/content/drive/MyDrive/furssah/evaluation_output/`

## Usage

1. **Data Preprocessing**:

   - Run `data_preprocessing.ipynb` to process 2,830,743 records, select 7 features, and sample 100,000 records.
   - Outputs include scaler, feature names, and train-test splits for training.
   - Use `preprocess_realtime_sample` for real-time flow preprocessing.

2. **Model Training**:

   - Run `models_training.ipynb` to train Random Forest models on 80,000 train/20,000 test samples per task.
   - Hyperparameter tuning uses GridSearchCV (24 combinations, 3-fold CV).
   - Outputs include trained models, performance metrics, visualizations, and feature importance.
   - Customize `param_grid` or `TRAINING_PATHS` for different configurations.

3. **Real-Time Evaluation**:

   - Run `realtime_evaluation.ipynb` to simulate real-time classification.
   - Load models, scaler, and feature names, then process sample flows.
   - Evaluate prediction latency and accuracy for deployment readiness.
   - Outputs include prediction results and latency visualizations.

4. **Folder Structure**:

   ```
   /content/drive/MyDrive/furssah/
   ├── MachineLearningCVE/              # Raw CICIDS2017 CSV files
   ├── data_preprocessing.ipynb
   ├── models_training.ipynb
   ├── realtime_evaluation.ipynb
   ├── preprocessing_output/            # Preprocessing artifacts
   │   ├── models/
   │   ├── features/
   │   ├── train_test_splits/
   │   ├── visualizations/
   │   └── encodings/
   ├── training_output/                # Training artifacts
   │   ├── trained_models/
   │   ├── results/
   │   ├── visualizations/
   │   └── feature_analysis/
   └── evaluation_output/              # Evaluation artifacts
       ├── predictions/
       └── visualizations/
   ```

## Notes

- **Dataset**: CICIDS2017 contains 2,830,743 records with 80 features, sampled to 100,000 for efficiency.
- **Performance**:
  - Models achieve \~0.98 accuracy/F1 across tasks, with `operator_type` performing best (Test F1: 0.9806).
  - Top features: `Bwd Packet Length Max`, `Total Length of Fwd Packets`, `Fwd Packets/s`.
- **Colab Performance**: CPU-based (\~2 cores, 13GB RAM), with training taking \~32.1 minutes (5.64x longer than estimated).
- **Real-Time Considerations**: Predictions should be low-latency (&lt;100ms) for deployment; test thoroughly with diverse flows.
- **Next Steps**:
  - Deploy models in a production environment for real-time threat detection.
  - Test with new network traffic data to ensure generalizability.
  - Explore ensemble methods or deep learning for improved performance.

## Example Outputs

- **Preprocessing**:
  - "Total combined dataset: 2,830,743 records, 80 features"
  - "Selected 7 features: \['Total Length of Fwd Packets', ...\]"
  - Artifacts saved in `/content/drive/MyDrive/furssah/preprocessing_output/`
- **Training**:
  - "Test Accuracy: 0.9801, Test F1: 0.9798 for attack_type"
  - "Top features: Bwd Packet Length Max (0.271), Total Length of Fwd Packets (0.254)"
  - Models and results saved in `/content/drive/MyDrive/furssah/training_output/`
- **Evaluation (Expected)**:
  - "Processed 1,000 flows in 85ms average latency"
  - "Real-time accuracy: 0.979, F1: 0.978"
  - Predictions saved in `/content/drive/MyDrive/furssah/evaluation_output/`

## License

This project is for educational and research purposes, utilizing the CICIDS2017 dataset. Ensure compliance with dataset usage policies and cite appropriately in academic work.

