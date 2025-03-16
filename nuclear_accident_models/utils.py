import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import glob
from tqdm import tqdm
import joblib

# Path configurations
OPERATION_DATA_DIR = "../Operation_csv_data"
OUTPUT_DIR = "../all_model_output"

def extract_features(df):
    """Extract features from time series data."""
    features = {}
    
    # Time and frequency domain features
    for column in df.columns:
        if column == 'TIME':
            continue
            
        values = df[column].values
        
        # Basic statistics
        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_median"] = np.median(values)
        features[f"{column}_range"] = np.max(values) - np.min(values)  # Added range
        
        # Rate of change features
        if len(values) > 1:
            diff = np.diff(values)
            features[f"{column}_diff_mean"] = np.mean(diff)
            features[f"{column}_diff_std"] = np.std(diff)
            features[f"{column}_diff_max"] = np.max(diff)
            features[f"{column}_diff_min"] = np.min(diff)  # Added min diff
            
            # Added more sophisticated features
            if len(values) > 2:
                # Second derivative (acceleration)
                accel = np.diff(diff)
                features[f"{column}_accel_mean"] = np.mean(accel)
                features[f"{column}_accel_std"] = np.std(accel)
                
            # Signal energy
            features[f"{column}_energy"] = np.sum(np.square(values)) / len(values)
            
            # Trend detection - linear fit slope
            x = np.arange(len(values))
            try:
                slope, _ = np.polyfit(x, values, 1)
                features[f"{column}_slope"] = slope
            except:
                features[f"{column}_slope"] = 0
                
    return features

def load_operation_data(accident_type, file_id):
    """Load operation data for a given accident type and file ID."""
    file_path = os.path.join(OPERATION_DATA_DIR, accident_type, f"{file_id}.csv")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def create_dataset():
    """Load all data and create a feature dataset for training."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all accident types
    accident_types = [d for d in os.listdir(OPERATION_DATA_DIR) 
                     if os.path.isdir(os.path.join(OPERATION_DATA_DIR, d)) and d != 'Normal']
    
    X = []  # Features
    y = []  # Labels
    file_info = []  # To keep track of which file each sample comes from
    
    # Process each accident type
    for accident_type in tqdm(accident_types, desc="Processing accident types"):
        print(f"Processing {accident_type}")
        
        # Get all CSV files for this accident type
        file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
        files = glob.glob(file_pattern)
        
        for file_path in tqdm(files, desc=f"Processing {accident_type} files"):
            file_id = os.path.basename(file_path).split('.')[0]
            
            # Load operation data
            operation_df = load_operation_data(accident_type, file_id)
            
            if operation_df is not None:
                # Extract features
                features = extract_features(operation_df)
                
                # Add to dataset
                X.append(features)
                y.append(accident_type)
                file_info.append({"accident_type": accident_type, "file_id": file_id})
    
    # Convert to DataFrame for easier handling
    X_df = pd.DataFrame(X)
    
    # Fill NaN values with 0
    X_df = X_df.fillna(0)
    
    return X_df, y, file_info

def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_dir=None):
    """Plot and save confusion matrix."""
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()
    
def plot_feature_importance(model, feature_names, model_name, output_dir=None):
    """Plot and save feature importance."""
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance if model has this attribute
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        return  # Model doesn't support feature importance
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_df.to_csv(os.path.join(output_dir, f"feature_importance_{model_name}.csv"), index=False)
    
    # Plot top 20 most important features
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'][:20][::-1], importance_df['importance'][:20][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Most Important Features - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{model_name}.png"))
    plt.close()

def save_model_results(model, X_test, y_test, y_pred, model_name, class_names, params=None, output_dir=None):
    """Save model, evaluation metrics, and visualizations."""
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    joblib.dump(model, os.path.join(output_dir, f"{model_name}.joblib"))
    
    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"classification_report_{model_name}.csv"))
    
    # Print the report
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names, model_name, output_dir)
    
    # Plot feature importance if applicable
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, X_test.columns, model_name, output_dir)
    
    # Save parameters if provided
    if params:
        with open(os.path.join(output_dir, f"params_{model_name}.txt"), 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for cls in set(y_test):
        cls_indices = [i for i, x in enumerate(y_test) if x == cls]
        if cls_indices:
            correct = sum(y_test[i] == y_pred[i] for i in cls_indices)
            class_accuracy[cls] = correct / len(cls_indices)
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for cls, acc in class_accuracy.items():
        print(f"{cls}: {acc:.4f}")
    
    return accuracy, class_accuracy 