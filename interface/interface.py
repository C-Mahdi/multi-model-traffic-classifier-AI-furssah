import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import time
from io import StringIO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Multi-Model Traffic Classifier",
    page_icon="🔒",
    layout="wide"
)

# Model configurations
MODEL_CONFIGS = {
    "VPN Classifier": {
        "model_file": "vpn_classifier_model.pkl",
        "model_type": "pickle",
        "data_format": "combined",
        "features": ['min_flowiat', 'max_flowiat', 'std_flowiat', 'flowBytesPerSecond', 'max_fiat'],
        "description": "Classifies VPN vs Non-VPN traffic",
        "classes": ["Non-VPN", "VPN"]
    },
    "Traffic Source": {
        "model_file": "traffic_source_random_forest_model.joblib",
        "model_type": "joblib",
        "data_format": "separate",
        "x_file": "traffic_source_X_test.csv",
        "y_file": "traffic_source_y_test.csv",
        "description": "Classifies traffic source types",
        "classes": []  # Will be auto-detected from data
    },
    "Operator Type": {
        "model_file": "operator_type_random_forest_model.joblib",
        "model_type": "joblib",
        "data_format": "separate",
        "x_file": "operator_type_X_test.csv",
        "y_file": "operator_type_y_test.csv",
        "description": "Classifies operator types",
        "classes": []  # Will be auto-detected from data
    },
    "Attack Type": {
        "model_file": "attack_type_random_forest_model.joblib",
        "model_type": "joblib",
        "data_format": "separate",
        "x_file": "attack_type_X_test.csv",
        "y_file": "attack_type_y_test.csv",
        "description": "Classifies different types of network attacks",
        "classes": []  # Will be auto-detected from data
    }
}

@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    config = MODEL_CONFIGS[model_name]
    try:
        if config["model_type"] == "pickle":
            with open(config["model_file"], 'rb') as file:
                model = pickle.load(file)
        else:  # joblib
            model = joblib.load(config["model_file"])
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{config['model_file']}' not found in the current directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def get_class_names(y_data, model_name):
    """Get class names from the data, handling both string and numeric labels"""
    config = MODEL_CONFIGS[model_name]
    
    # Get unique classes from the data
    unique_classes = sorted(list(set(y_data)))
    
    # If model has predefined classes, use them
    if config["classes"] and len(config["classes"]) > 0:
        return config["classes"], unique_classes
    
    # Otherwise, use the actual unique values as class names
    class_names = [str(cls) for cls in unique_classes]
    return class_names, unique_classes

def extract_features_and_labels(df, model_name, label_column=None):
    """Extract features (X) and labels (y) from the dataframe based on model type"""
    config = MODEL_CONFIGS[model_name]
    
    if model_name == "VPN Classifier":
        # Original VPN classifier logic
        expected_features = config["features"]
        
        # Check if all expected features exist
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        if missing_features:
            return None, None, f"Missing required features: {missing_features}"
        
        # Extract features
        X = df[expected_features]
        
        # Extract labels
        if label_column:
            if label_column not in df.columns:
                return None, None, f"Label column '{label_column}' not found in data"
            y = df[label_column]
        else:
            # Try to find label column automatically
            possible_label_columns = ['label', 'target', 'y', 'class', 'classification']
            label_col = None
            
            for col in possible_label_columns:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                # Use the last column as label
                label_col = df.columns[-1]
                st.warning(f"⚠️ No standard label column found. Using '{label_col}' as label column.")
            
            y = df[label_col]
        
        return X, y, None
    
    else:
        # For other models, assume all columns except label are features
        if label_column and label_column in df.columns:
            feature_columns = [col for col in df.columns if col != label_column]
            X = df[feature_columns]
            y = df[label_col]
        else:
            # Auto-detect label column
            possible_label_columns = ['label', 'target', 'y', 'class', 'classification']
            label_col = None
            
            for col in possible_label_columns:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                label_col = df.columns[-1]
                st.warning(f"⚠️ Using '{label_col}' as label column.")
            
            feature_columns = [col for col in df.columns if col != label_col]
            X = df[feature_columns]
            y = df[label_col]
        
        return X, y, None

def load_separate_datasets(x_file, y_file):
    """Load separate X and Y datasets"""
    try:
        X = pd.read_csv(x_file)
        y = pd.read_csv(y_file)
        
        # If y has multiple columns, take the first one or find the target column
        if y.shape[1] > 1:
            # Try to find a target column
            target_cols = ['label', 'target', 'y', 'class', 'classification']
            target_col = None
            for col in target_cols:
                if col in y.columns:
                    target_col = col
                    break
            
            if target_col:
                y = y[target_col]
            else:
                y = y.iloc[:, 0]  # Take first column
                st.warning(f"Multiple columns in Y file. Using '{y.name}' as target.")
        else:
            y = y.iloc[:, 0]
        
        return X, y, None
    except Exception as e:
        return None, None, f"Error loading datasets: {str(e)}"

def batch_evaluation_with_comparison(model, X_test, y_test, model_name, batch_size=1000, warmup_runs=2, progress_container=None):
    """Evaluate model in batches and compare with real outputs"""
    # Get class names from actual data
    class_names, unique_classes = get_class_names(y_test, model_name)
    
    total_samples = len(X_test)
    num_batches = int(np.ceil(total_samples / batch_size))
    
    # Initialize containers for real-time updates
    if progress_container:
        status_placeholder = progress_container.empty()
        current_batch_placeholder = progress_container.empty()
        summary_placeholder = progress_container.empty()
        chart_placeholder = progress_container.empty()
    
    # Initialize tracking variables
    batch_results = []
    all_predictions = []
    all_true_labels = []
    total_prediction_time = 0
    
    # Warmup phase
    if progress_container:
        status_placeholder.info(f"🔥 Warming up {model_name} model with {warmup_runs} runs...")
    
    warmup_batch = X_test[:min(batch_size, len(X_test))]
    for _ in range(warmup_runs):
        _ = model.predict(warmup_batch)
    
    if progress_container:
        status_placeholder.success("✅ Warmup completed! Starting batch evaluation...")
    
    # Progress bar
    progress_bar = st.progress(0) if progress_container else None
    
    # Main evaluation loop
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        # Get batch data
        batch_X = X_test[start_idx:end_idx]
        batch_y_true = y_test[start_idx:end_idx]
        current_batch_size = len(batch_X)
        
        # Time the prediction
        start_time = time.time()
        batch_predictions = model.predict(batch_X)
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_prediction_time += batch_time
        
        # Calculate batch metrics
        batch_accuracy = accuracy_score(batch_y_true, batch_predictions)
        batch_precision = precision_score(batch_y_true, batch_predictions, average='weighted', zero_division=0)
        batch_recall = recall_score(batch_y_true, batch_predictions, average='weighted', zero_division=0)
        batch_f1 = f1_score(batch_y_true, batch_predictions, average='weighted', zero_division=0)
        
        # Time metrics
        time_per_sample = batch_time / current_batch_size
        samples_per_second = current_batch_size / batch_time
        
        # Get unique classes for this batch
        batch_unique_classes = np.unique(np.concatenate([batch_y_true, batch_predictions]))
        
        # Store batch results
        batch_result = {
            'batch_number': batch_idx + 1,
            'start_row': start_idx + 1,
            'end_row': end_idx,
            'batch_size': current_batch_size,
            'accuracy': batch_accuracy,
            'precision': batch_precision,
            'recall': batch_recall,
            'f1_score': batch_f1,
            'prediction_time': batch_time,
            'time_per_sample': time_per_sample,
            'samples_per_second': samples_per_second,
        }
        
        # Add class-specific metrics for binary classification
        if len(unique_classes) == 2:
            # For binary classification, use the second class as positive
            positive_class = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            negative_class = unique_classes[0] if len(unique_classes) > 1 else unique_classes[0]
            
            batch_result.update({
                'true_positives': np.sum((batch_y_true == positive_class) & (batch_predictions == positive_class)),
                'true_negatives': np.sum((batch_y_true == negative_class) & (batch_predictions == negative_class)),
                'false_positives': np.sum((batch_y_true == negative_class) & (batch_predictions == positive_class)),
                'false_negatives': np.sum((batch_y_true == positive_class) & (batch_predictions == negative_class)),
                'positive_predicted': np.sum(batch_predictions == positive_class),
                'positive_actual': np.sum(batch_y_true == positive_class)
            })
        
        # Add class distribution for multiclass
        for i, class_val in enumerate(unique_classes):
            class_name = class_names[i] if i < len(class_names) else str(class_val)
            batch_result[f'{class_name}_predicted'] = np.sum(batch_predictions == class_val)
            batch_result[f'{class_name}_actual'] = np.sum(batch_y_true == class_val)
        
        batch_results.append(batch_result)
        
        # Store all predictions and true labels for global analysis
        all_predictions.extend(batch_predictions)
        all_true_labels.extend(batch_y_true)
        
        # Update progress
        if progress_bar:
            progress_bar.progress((batch_idx + 1) / num_batches)
        
        # Real-time batch display
        if progress_container and current_batch_placeholder:
            with current_batch_placeholder.container():
                st.subheader(f"📊 Batch {batch_idx + 1}/{num_batches} Results - {model_name}")
                
                # Batch metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{batch_accuracy:.3f}")
                with col2:
                    st.metric("Precision", f"{batch_precision:.3f}")
                with col3:
                    st.metric("Recall", f"{batch_recall:.3f}")
                with col4:
                    st.metric("F1-Score", f"{batch_f1:.3f}")
                with col5:
                    st.metric("Time", f"{batch_time:.4f}s")
                
                # Timing metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Time/Sample", f"{time_per_sample:.6f}s")
                with col2:
                    st.metric("Samples/Second", f"{samples_per_second:.1f}")
                with col3:
                    st.metric("Rows", f"{start_idx + 1}-{end_idx}")
                
                # Class-specific metrics
                if len(unique_classes) == 2:
                    # Binary classification
                    col1, col2 = st.columns(2)
                    with col1:
                        positive_class_name = class_names[1] if len(class_names) > 1 else str(unique_classes[1])
                        st.metric(f"{positive_class_name} Predicted", batch_result['positive_predicted'])
                    with col2:
                        st.metric(f"{positive_class_name} Actual", batch_result['positive_actual'])
                else:
                    # Multiclass classification
                    cols = st.columns(min(len(unique_classes), 4))
                    for i, (class_val, col) in enumerate(zip(unique_classes[:4], cols)):
                        class_name = class_names[i] if i < len(class_names) else str(class_val)
                        with col:
                            predicted_key = f'{class_name}_predicted'
                            actual_key = f'{class_name}_actual'
                            if predicted_key in batch_result:
                                st.metric(f"{class_name}", f"P:{batch_result[predicted_key]} A:{batch_result[actual_key]}")
        
        # Update running summary every 5 batches or on last batch
        if progress_container and summary_placeholder and (batch_idx % 5 == 0 or batch_idx == num_batches - 1):
            with summary_placeholder.container():
                st.subheader(f"📈 Running Summary - {model_name}")
                
                # Calculate running metrics
                processed_samples = sum([r['batch_size'] for r in batch_results])
                running_accuracy = accuracy_score(all_true_labels, all_predictions)
                avg_time_per_sample = total_prediction_time / processed_samples
                total_samples_per_second = processed_samples / total_prediction_time
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Processed Samples", f"{processed_samples}/{total_samples}")
                with col2:
                    st.metric("Running Accuracy", f"{running_accuracy:.3f}")
                with col3:
                    st.metric("Avg Time/Sample", f"{avg_time_per_sample:.6f}s")
                with col4:
                    st.metric("Overall Throughput", f"{total_samples_per_second:.1f} samples/s")
        
        # Update charts
        if progress_container and chart_placeholder and len(batch_results) > 1:
            df_batches = pd.DataFrame(batch_results)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy per Batch', 'Timing per Batch', 'Class Distribution', 'Performance Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(x=df_batches['batch_number'], y=df_batches['accuracy'], 
                          name='Accuracy', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Timing plot
            fig.add_trace(
                go.Scatter(x=df_batches['batch_number'], y=df_batches['samples_per_second'], 
                          name='Samples/Second', line=dict(color='green')),
                row=1, col=2
            )
            
            # Class distribution plot
            if len(unique_classes) == 2:
                positive_class_name = class_names[1] if len(class_names) > 1 else str(unique_classes[1])
                fig.add_trace(
                    go.Scatter(x=df_batches['batch_number'], y=df_batches['positive_predicted'], 
                              name=f'{positive_class_name} Predicted', line=dict(color='red')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df_batches['batch_number'], y=df_batches['positive_actual'], 
                              name=f'{positive_class_name} Actual', line=dict(color='orange')),
                    row=2, col=1
                )
            else:
                # For multiclass, show first two classes
                for i, class_val in enumerate(unique_classes[:2]):
                    class_name = class_names[i] if i < len(class_names) else str(class_val)
                    predicted_key = f'{class_name}_predicted'
                    actual_key = f'{class_name}_actual'
                    if predicted_key in df_batches.columns:
                        colors = ['red', 'blue']
                        fig.add_trace(
                            go.Scatter(x=df_batches['batch_number'], y=df_batches[predicted_key], 
                                      name=f'{class_name} Pred', line=dict(color=colors[i])),
                            row=2, col=1
                        )
            
            # Performance metrics
            fig.add_trace(
                go.Scatter(x=df_batches['batch_number'], y=df_batches['precision'], 
                          name='Precision', line=dict(color='purple')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=df_batches['batch_number'], y=df_batches['recall'], 
                          name='Recall', line=dict(color='brown')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=df_batches['batch_number'], y=df_batches['f1_score'], 
                          name='F1-Score', line=dict(color='pink')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, title=f"{model_name} - Real-time Performance")
            fig.update_xaxes(title_text="Batch Number")
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Small delay for visual effect
        time.sleep(0.1)
    
    # Calculate global metrics
    global_accuracy = accuracy_score(all_true_labels, all_predictions)
    global_precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    global_recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    global_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    
    return {
        'batch_results': batch_results,
        'global_metrics': {
            'total_samples': total_samples,
            'total_prediction_time': total_prediction_time,
            'avg_time_per_sample': total_prediction_time / total_samples,
            'total_throughput': total_samples / total_prediction_time,
            'global_accuracy': global_accuracy,
            'global_precision': global_precision,
            'global_recall': global_recall,
            'global_f1': global_f1
        },
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'confusion_matrix': confusion_matrix(all_true_labels, all_predictions),
        'class_names': class_names,
        'unique_classes': unique_classes
    }

# Title and description
st.title("🔒 Multi-Model Traffic Classifier")
st.markdown("Comprehensive evaluation system for multiple machine learning models with different data formats.")

# Model selection
st.header("🎯 Model Selection")
selected_model = st.selectbox(
    "Choose a model to evaluate:",
    list(MODEL_CONFIGS.keys()),
    help="Select the model you want to evaluate"
)

# Display selected model info
config = MODEL_CONFIGS[selected_model]
st.info(f"**{selected_model}**: {config['description']}")

# Load model
model, error_message = load_model(selected_model)

# Display model status
if model is not None:
    st.success(f"✅ {selected_model} model loaded successfully!")
else:
    st.error(f"❌ {error_message}")
    st.stop()

# Sidebar for model information
st.sidebar.header("Model Information")
st.sidebar.markdown(f"""
**Selected Model:** {selected_model}
**Type:** {config['model_type'].upper()}
**Data Format:** {config['data_format'].title()}
**Description:** {config['description']}
""")

if config['data_format'] == 'combined':
    st.sidebar.markdown(f"""
**Required Features:**
{chr(10).join([f"- {feat}" for feat in config['features']])}
""")

# Data upload section based on model type
st.header("📁 Data Upload")

if config['data_format'] == 'combined':
    # Single file upload for VPN classifier
    st.markdown(f"Upload a single CSV file containing both features and labels for **{selected_model}**")
    uploaded_file = st.file_uploader(
        "Choose CSV file with features + labels",
        type="csv",
        help="Upload CSV file containing both features and labels",
        key="combined_upload"
    )
    
    # Label column selection
    label_column = st.selectbox(
        "Select label column (or leave empty for auto-detection)",
        ["Auto-detect"],
        help="Choose the column containing the true labels"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Extract features and labels
            label_col = None if label_column == "Auto-detect" else label_column
            X, y, error = extract_features_and_labels(df, selected_model, label_col)
            
            if error:
                st.error(f"❌ {error}")
                st.info("Available columns: " + ", ".join(df.columns))
                X, y = None, None
            else:
                st.success("✅ Successfully extracted features and labels!")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            X, y = None, None
    else:
        X, y = None, None

else:
    # Separate file upload for other models
    st.markdown(f"Upload separate X and Y files for **{selected_model}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_file = st.file_uploader(
            "Choose X (features) CSV file",
            type="csv",
            help="Upload CSV file containing features",
            key="x_upload"
        )
    
    with col2:
        y_file = st.file_uploader(
            "Choose Y (labels) CSV file", 
            type="csv",
            help="Upload CSV file containing labels",
            key="y_upload"
        )
    
    if x_file is not None and y_file is not None:
        try:
            # Load separate datasets
            X, y, error = load_separate_datasets(x_file, y_file)
            
            if error:
                st.error(f"❌ {error}")
                X, y = None, None
            else:
                st.success("✅ Successfully loaded X and Y datasets!")
        
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            X, y = None, None
    else:
        X, y = None, None
        if x_file is None and y_file is None:
            st.info("👆 Please upload both X and Y CSV files to start evaluation.")
        elif x_file is None:
            st.warning("⚠️ Please upload the X (features) CSV file.")
        else:
            st.warning("⚠️ Please upload the Y (labels) CSV file.")

# Continue with evaluation if data is loaded
if X is not None and y is not None:
    # Data overview
    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(X))
    with col2:
        st.metric("Feature Columns", len(X.columns))
    with col3:
        st.metric("Missing Values", X.isnull().sum().sum() + y.isnull().sum())
    
    # Display data preview
    st.subheader("📋 Data Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Features (X)")
        st.dataframe(X.head(), use_container_width=True)
        st.info(f"Shape: {X.shape}")
    
    with col2:
        st.subheader("🎯 Labels (y)")
        st.dataframe(y.head().to_frame() if hasattr(y, 'to_frame') else pd.DataFrame(y).head(), use_container_width=True)
        st.info(f"Shape: {y.shape if hasattr(y, 'shape') else (len(y),)}")
        
        # Show label distribution
        if hasattr(y, 'value_counts'):
            label_counts = y.value_counts().to_dict()
        else:
            unique, counts = np.unique(y, return_counts=True)
            label_counts = dict(zip(unique, counts))
        
        st.write("Label distribution:", label_counts)
        
        # Calculate class percentages
        total_samples = len(y)
        for class_val, count in label_counts.items():
            percentage = (count / total_samples) * 100
            st.metric(f"{str(class_val)} %", f"{percentage:.1f}%")
    
    # Feature statistics
    st.subheader("📈 Feature Statistics")
    st.dataframe(X.describe(), use_container_width=True)
    
    # Evaluation parameters
    st.subheader("⚙️ Evaluation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input("Batch Size", min_value=100, max_value=5000, value=1000, step=100)
    with col2:
        warmup_runs = st.slider("Warmup Runs", min_value=1, max_value=5, value=2)
    
    # Calculate number of batches
    num_batches = int(np.ceil(len(X) / batch_size))
    
    st.subheader("📊 Evaluation Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Batch Size", batch_size)
    with col3:
        st.metric("Number of Batches", num_batches)
    
    # Start evaluation button
    if st.button("🚀 Start Batch Evaluation", type="primary", key="eval_button"):
        st.header(f"📈 Real-Time Batch Evaluation - {selected_model}")
        
        # Create container for real-time updates
        progress_container = st.container()
        
        # Convert to numpy arrays for evaluation
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Run evaluation
        results = batch_evaluation_with_comparison(
            model, X_array, y_array, selected_model, batch_size, warmup_runs, progress_container
        )
        
        # Display final results
        st.header(f"🎯 Final Global Report - {selected_model}")
        
        # Global metrics
        global_metrics = results['global_metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Global Accuracy", f"{global_metrics['global_accuracy']:.4f}")
        with col2:
            st.metric("Global Precision", f"{global_metrics['global_precision']:.4f}")
        with col3:
            st.metric("Global Recall", f"{global_metrics['global_recall']:.4f}")
        with col4:
            st.metric("Global F1-Score", f"{global_metrics['global_f1']:.4f}")
        
        # Timing metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{global_metrics['total_prediction_time']:.4f}s")
        with col2:
            st.metric("Avg Time/Sample", f"{global_metrics['avg_time_per_sample']:.6f}s")
        with col3:
            st.metric("Total Throughput", f"{global_metrics['total_throughput']:.1f} samples/s")
        
        # Detailed batch results table
        st.subheader("📋 Detailed Batch Results")
        batch_df = pd.DataFrame(results['batch_results'])
        
        # Format the dataframe for better display
        display_df = batch_df.copy()
        numeric_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'prediction_time', 'time_per_sample', 'samples_per_second']
        for col in numeric_columns:
            if col in display_df.columns:
                if col == 'time_per_sample':
                    display_df[col] = display_df[col].round(6)
                elif col == 'samples_per_second':
                    display_df[col] = display_df[col].round(1)
                elif col == 'prediction_time':
                    display_df[col] = display_df[col].round(4)
                else:
                    display_df[col] = display_df[col].round(4)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Global Confusion Matrix
        st.subheader("🔍 Global Confusion Matrix")
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        # Create confusion matrix heatmap
        fig_cm = px.imshow(cm, 
                         text_auto=True, 
                         aspect="auto",
                         labels=dict(x="Predicted", y="Actual"),
                         x=class_names[:cm.shape[1]],
                         y=class_names[:cm.shape[0]],
                         color_continuous_scale='Blues')
        fig_cm.update_layout(title=f"Global Confusion Matrix - {selected_model}")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Global Classification Report")
        try:
            report = classification_report(
                results['all_true_labels'], 
                results['all_predictions'], 
                target_names=class_names, 
                output_dict=True
            )
            
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        except Exception as e:
            st.error(f"Error generating classification report: {str(e)}")
            st.text(classification_report(results['all_true_labels'], results['all_predictions']))
        
        # Download options
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download batch results
            batch_csv = batch_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Batch Results",
                data=batch_csv,
                file_name=f"{selected_model.lower().replace(' ', '_')}_batch_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download global summary
            global_summary = pd.DataFrame([global_metrics])
            global_csv = global_summary.to_csv(index=False)
            st.download_button(
                label="🌍 Download Global Summary",
                data=global_csv,
                file_name=f"{selected_model.lower().replace(' ', '_')}_global_summary.csv",
                mime="text/csv"
            )

# Instructions section
st.header("📝 Instructions")
st.markdown(f"""
## Multi-Model Evaluation Process

### Available Models:
{chr(10).join([f"- **{name}**: {config['description']}" for name, config in MODEL_CONFIGS.items()])}

### Data Format Requirements:

#### VPN Classifier (Combined Format):
- Single CSV file with features and labels
- Required features: min_flowiat, max_flowiat, std_flowiat, flowBytesPerSecond, max_fiat
- Label column: any column with 0/1 values

#### Other Models (Separate Format):
- Two CSV files: X (features) and Y (labels)
- X file: contains all feature columns
- Y file: contains target/label column
- **Supports both string and numeric labels** (e.g., 'BENIGN', 'MALICIOUS' or 0, 1)

### Evaluation Process:
1. **Select Model**: Choose from available models
2. **Upload Data**: Upload required files based on model type
3. **Set Parameters**: Adjust batch size and warmup runs
4. **Start Evaluation**: Click "Start Batch Evaluation"
5. **Monitor Progress**: Watch real-time metrics
6. **Download Results**: Save detailed reports

### What You'll Get:
- **Per-batch metrics**: Accuracy, Precision, Recall, F1-Score
- **Performance metrics**: Prediction time, throughput
- **Real-time charts**: Live performance visualization
- **Global report**: Overall model performance
- **Confusion matrix**: Detailed classification results with actual class names
- **Downloadable results**: CSV reports for further analysis

### Supported Label Types:
- **Numeric**: 0, 1, 2, etc.
- **String**: 'BENIGN', 'MALICIOUS', 'VPN', 'Non-VPN', etc.
- **Mixed**: The system automatically detects and handles different label formats
""")

# Footer
st.markdown("---")
st.markdown(" Multi-Model Traffic Classifier")
