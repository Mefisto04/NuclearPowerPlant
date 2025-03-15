import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import glob
from tqdm import tqdm
import seaborn as sns
from collections import Counter

# Try to import SMOTE for handling imbalanced classes
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imblearn not installed. SMOTE oversampling will not be available.")
    print("To install it, run: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

# Path configurations
OPERATION_DATA_DIR = "Operation_csv_data"
DOSE_DATA_DIR = "Dose_csv_data" 
TRANSIENT_DATA_DIR = "NPPAD"
OUTPUT_DIR = "model_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to extract features from time series data
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
        
        # Rate of change features
        if len(values) > 1:
            diff = np.diff(values)
            features[f"{column}_diff_mean"] = np.mean(diff)
            features[f"{column}_diff_std"] = np.std(diff)
            features[f"{column}_diff_max"] = np.max(diff)
            
    return features

# Function to load and preprocess a dataset
def load_operation_data(accident_type, file_id):
    """Load operation data for a given accident type and file ID."""
    file_path = os.path.join(OPERATION_DATA_DIR, accident_type, f"{file_id}.csv")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to load all data and create a feature dataset
def create_dataset():
    """Load all data and create a feature dataset for training."""
    
    # Get all accident types
    accident_types = [d for d in os.listdir(OPERATION_DATA_DIR) 
                     if os.path.isdir(os.path.join(OPERATION_DATA_DIR, d)) and d != 'Normal']
    
    X = []  # Features
    y = []  # Labels
    
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
                
    # Convert to DataFrame for easier handling
    X_df = pd.DataFrame(X)
    
    # Fill NaN values with 0
    X_df = X_df.fillna(0)
    
    return X_df, y

def train_model_with_cross_validation(X, y):
    """Train and evaluate a classification model using cross-validation."""
    
    print("Class distribution in full dataset:")
    class_counts = pd.Series(y).value_counts()
    print(class_counts)
    
    min_samples = class_counts.min()
    
    # If any class has only 1 sample, we can't do standard cross-validation
    if min_samples < 2:
        print(f"\nWARNING: Some classes have only {min_samples} sample(s). Cannot perform standard cross-validation.")
        print("Using a modified validation approach for rare classes...")
        
        # Identify rare and common classes
        rare_classes = class_counts[class_counts < 2].index.tolist()
        common_classes = class_counts[class_counts >= 2].index.tolist()
        
        print(f"Rare classes (only 1 sample): {rare_classes}")
        print(f"Common classes (2+ samples): {common_classes}")
        
        # Create indices for rare and common samples
        rare_indices = [i for i, label in enumerate(y) if label in rare_classes]
        common_indices = [i for i, label in enumerate(y) if label in common_classes]
        
        # Convert to numpy arrays for easier handling
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = np.array(y)
        
        # For common classes, use stratified CV
        X_common = X.iloc[common_indices] if isinstance(X, pd.DataFrame) else X[common_indices]
        y_common = [y[i] for i in common_indices]
        
        # Use 3-fold CV for common classes
        n_splits = 3
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        print(f"Using {n_splits}-fold CV for common classes and holding out rare classes")
        
        # Define model pipeline with regularization to prevent overfitting
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,  # Add regularization
                min_samples_split=5,  # More regularization
                min_samples_leaf=2,   # Additional regularization
                random_state=42
            ))
        ])
        
        # Train on common classes
        print("Training on common classes...")
        pipeline.fit(X_common, y_common)
        
        # Evaluate on rare classes (as a special validation set)
        if rare_indices:
            print("Evaluating on rare classes...")
            X_rare = X.iloc[rare_indices] if isinstance(X, pd.DataFrame) else X[rare_indices]
            y_rare = [y[i] for i in rare_indices]
            y_rare_pred = pipeline.predict(X_rare)
            
            rare_accuracy = sum(y_rare_pred[i] == y_rare[i] for i in range(len(y_rare))) / len(y_rare)
            print(f"Rare class prediction accuracy: {rare_accuracy:.4f}")
            
            for i, (true, pred) in enumerate(zip(y_rare, y_rare_pred)):
                print(f"  Rare class {rare_classes[i]}: True={true}, Predicted={pred}")
        
        # Perform cross-validation on common classes
        cv_scores = cross_val_score(pipeline, X_common, y_common, cv=cv, scoring='accuracy')
        print(f"Common classes cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Refit on all data
        print("Fitting final model on all data...")
        pipeline.fit(X, y)
        
        # Get best parameters
        best_params = pipeline.get_params()
        important_params = {
            'classifier__n_estimators': best_params['classifier__n_estimators'],
            'classifier__max_depth': best_params['classifier__max_depth'],
            'classifier__min_samples_split': best_params['classifier__min_samples_split'],
            'classifier__min_samples_leaf': best_params['classifier__min_samples_leaf']
        }
        
        # Save the model
        joblib.dump(pipeline, os.path.join(OUTPUT_DIR, "nuclear_accident_classifier.joblib"))
        
        return pipeline, important_params, cv_scores
    
    # Standard cross-validation for datasets where all classes have at least 2 samples
    else:
        print("\nUsing cross-validation to evaluate model performance...")
        
        # Define number of folds
        n_splits = min(5, min_samples)
        print(f"Using standard {n_splits}-fold CV")
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Define model pipeline with regularization
        if SMOTE_AVAILABLE:
            # Pipeline with SMOTE for handling imbalanced classes
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        else:
            # Standard pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        
        # Define hyperparameters for grid search with regularization
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],  # Added regularization
            'classifier__min_samples_split': [2, 5, 10],  # More regularization
            'classifier__min_samples_leaf': [1, 2, 4]  # Additional regularization
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        # Get cross-validation results
        cv_results = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
        print(f"\nCross-validation accuracy: {cv_results.mean():.4f} (±{cv_results.std():.4f})")
        
        # Fit the model on the full dataset
        best_model.fit(X, y)
        
        # Save the model
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, "nuclear_accident_classifier.joblib"))
        
        return best_model, best_params, cv_results

def train_model(X, y):
    """Train a classification model with standard train-test split for comparison."""
    
    # Check if all classes have at least 2 samples for stratification
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    min_samples = class_counts.min()
    
    # Split data into train and test sets
    if min_samples >= 2:
        # If all classes have at least 2 samples, use stratification
        print(f"Using stratified split (min_samples per class: {min_samples})")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # If some classes have only 1 sample, don't use stratification
        print(f"Warning: Some classes have only {min_samples} sample(s). Using regular split without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print class distribution in train and test sets
    print("Training set class distribution:")
    print(pd.Series(y_train).value_counts())
    print("Test set class distribution:")
    print(pd.Series(y_test).value_counts())
    
    # Apply SMOTE to handle imbalanced classes if available
    if SMOTE_AVAILABLE:
        print("\nApplying SMOTE to balance training data...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1) if min_samples > 1 else 1)
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print("After SMOTE, training set class distribution:")
            print(pd.Series(y_train_resampled).value_counts())
        except ValueError as e:
            print(f"SMOTE failed: {e}. Using original imbalanced data.")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Define the model pipeline with regularization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,  # Add some regularization
            min_samples_split=5,  # More regularization 
            min_samples_leaf=2,  # Additional regularization
            random_state=42
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y)), 
                yticklabels=sorted(set(y)))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    
    # Save the model
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, "nuclear_accident_classifier_traditional.joblib"))
    
    # Get feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    # Plot top 20 most important features
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'][:20][::-1], importance_df['importance'][:20][::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    
    return pipeline, X_test, y_test, y_pred

def predict_accident_type(model, data_path):
    """Predict the accident type for new data."""
    try:
        # Load the data
        df = pd.read_csv(data_path)
        
        # Extract features
        features = extract_features(df)
        features_df = pd.DataFrame([features])
        
        # Fill NaN values with 0
        features_df = features_df.fillna(0)
        
        # Make sure the feature columns match the training data
        missing_cols = set(model.feature_names_in_) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0
            
        # Ensure columns are in the same order as during training
        features_df = features_df[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
        
        # Get top 3 most likely accident types
        class_indices = np.argsort(probabilities[0])[::-1][:3]
        top_classes = [model.classes_[i] for i in class_indices]
        top_probs = [probabilities[0][i] for i in class_indices]
        
        return {
            'predicted_type': prediction[0],
            'top_3_types': top_classes,
            'top_3_probs': top_probs
        }
        
    except Exception as e:
        print(f"Error predicting accident type: {e}")
        return None

def main():
    print("Creating dataset...")
    X, y = create_dataset()
    
    print("Dataset created with shape:", X.shape)
    print("Number of accident types:", len(set(y)))
    print("Accident types:", sorted(set(y)))
    
    print("\n==================================================")
    print("APPROACH 1: TRADITIONAL TRAIN-TEST SPLIT")
    print("==================================================")
    print("Training model with traditional train-test split...")
    traditional_model, X_test, y_test, y_pred = train_model(X, y)
    
    print("\n==================================================")
    print("APPROACH 2: CROSS-VALIDATION EVALUATION")
    print("==================================================")
    print("Training and evaluating model with cross-validation...")
    cv_model, best_params, cv_results = train_model_with_cross_validation(X, y)
    
    print("\nTraditional model accuracy:", accuracy_score(y_test, y_pred))
    print(f"Cross-validation model mean accuracy: {cv_results.mean():.4f} (±{cv_results.std():.4f})")
    
    # Compare the models
    print("\nComparing model parameters:")
    print("Traditional model parameters:")
    trad_params = traditional_model.get_params()
    print(f"  max_depth: {trad_params['classifier__max_depth']}")
    print(f"  min_samples_split: {trad_params['classifier__min_samples_split']}")
    print(f"  min_samples_leaf: {trad_params['classifier__min_samples_leaf']}")
    
    print("Cross-validated model best parameters:")
    for param, value in best_params.items():
        if param.startswith('classifier__'):
            print(f"  {param.replace('classifier__', '')}: {value}")
    
    # Save class labels
    with open(os.path.join(OUTPUT_DIR, "accident_types.txt"), 'w') as f:
        for accident_type in sorted(set(y)):
            f.write(f"{accident_type}\n")
    
    print("\nModel and related files saved to:", OUTPUT_DIR)
    
    # Example prediction
    print("\nExample prediction:")
    example_file = os.path.join(OPERATION_DATA_DIR, list(set(y))[0], "1.csv")
    prediction = predict_accident_type(cv_model, example_file)
    
    if prediction:
        print(f"Predicted accident type: {prediction['predicted_type']}")
        print("Top 3 most likely accident types:")
        for i in range(3):
            print(f"  {prediction['top_3_types'][i]}: {prediction['top_3_probs'][i]:.4f}")

if __name__ == "__main__":
    main() 