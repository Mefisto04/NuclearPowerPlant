"""
MODEL 3: ENSEMBLE OF MULTIPLE CLASSIFIERS

This model addresses the challenges in the original implementation by:
1. Creating an ensemble of diverse classifiers (Random Forest, XGBoost, LightGBM)
2. Using a voting mechanism to combine predictions from different models
3. Implementing a tiered approach for rare classes vs. common classes
4. Fine-tuning each model separately for its specialized task
5. Using different sampling strategies for different base learners

This ensemble approach leverages the strengths of different algorithms
to improve prediction for rare classes while maintaining accuracy on common ones.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_accident_models.utils import create_dataset, save_model_results

# Try to import SMOTE for handling imbalanced classes
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imblearn not installed. SMOTE oversampling will not be available.")
    print("To install it, run: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

def train_ensemble_model():
    """Train an ensemble of multiple classifiers for improved performance"""
    
    # Create model-specific output directory
    model_output_dir = "../all_model_output/ensemble"
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("Creating dataset...")
    X, y, file_info = create_dataset()
    
    print("Dataset created with shape:", X.shape)
    print("Number of accident types:", len(set(y)))
    print("Accident types:", sorted(set(y)))
    
    # Group the accident types into 'rare' and 'common' categories
    class_counts = pd.Series(y).value_counts()
    rare_threshold = 5  # Classes with <= 5 samples are considered rare
    
    rare_classes = class_counts[class_counts <= rare_threshold].index.tolist()
    common_classes = class_counts[class_counts > rare_threshold].index.tolist()
    
    print(f"Rare classes (≤{rare_threshold} samples): {rare_classes}")
    print(f"Common classes (>{rare_threshold} samples): {common_classes}")
    
    # Create a new feature indicating if the accident is rare
    is_rare = [1 if cls in rare_classes else 0 for cls in y]
    
    # Split the data while trying to maintain rare classes in both sets
    X_train, X_test, y_train, y_test, is_rare_train, is_rare_test = train_test_split(
        X, y, is_rare, test_size=0.2, random_state=42
    )
    
    # Check distribution of classes in train and test sets
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    
    print("\nClass distribution in training set:")
    print(train_counts)
    
    print("\nClass distribution in test set:")
    print(test_counts)
    
    # Check if we have all rare classes in both train and test sets
    missing_in_train = set(rare_classes) - set(y_train)
    missing_in_test = set(rare_classes) - set(y_test)
    
    if missing_in_train:
        print(f"\nWarning: Rare classes missing from training set: {missing_in_train}")
    
    if missing_in_test:
        print(f"\nWarning: Rare classes missing from test set: {missing_in_test}")
    
    # Initialize our ensemble components
    classifiers = []
    
    # 1. Random Forest - Good for overall structure of the data
    rf_params = {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'class_weight': 'balanced_subsample',
        'n_jobs': -1,
        'random_state': 42
    }
    
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**rf_params))
    ])
    
    classifiers.append(('rf', rf_pipeline))
    
    # 2. XGBoost - Good for complex non-linear relationships
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',
        'random_state': 42
    }
    
    # Calculate scale_pos_weight for XGBoost
    # This parameter helps with class imbalance
    xgb_class_weights = {}
    for cls in set(y_train):
        count = list(y_train).count(cls)
        xgb_class_weights[cls] = len(y_train) / (len(set(y_train)) * count)
    
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ])
    
    classifiers.append(('xgb', xgb_pipeline))
    
    # 3. LightGBM - Fast and efficient, good for high-dimensional data
    lgb_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'random_state': 42
    }
    
    lgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(**lgb_params))
    ])
    
    classifiers.append(('lgb', lgb_pipeline))
    
    # Create a specialized model for rare classes if we have any
    if rare_classes and SMOTE_AVAILABLE:
        print("\nCreating specialized model for rare classes using SMOTE...")
        
        # Get indices of samples with common and rare classes
        common_indices = [i for i, label in enumerate(y_train) if label in common_classes]
        rare_indices = [i for i, label in enumerate(y_train) if label in rare_classes]
        
        # Extract rare class samples
        X_rare = X_train.iloc[rare_indices]
        y_rare = [y_train[i] for i in rare_indices]
        
        # Add some common class samples to teach the model the difference
        # This is to avoid the rare model always predicting rare classes
        # We'll include fewer common samples to maintain focus on rare classes
        common_sample_indices = np.random.choice(common_indices, min(len(common_indices), len(rare_indices) * 3), replace=False)
        X_rare_plus = pd.concat([X_rare, X_train.iloc[common_sample_indices]])
        y_rare_plus = np.concatenate([y_rare, [y_train[i] for i in common_sample_indices]])
        
        # Try different SMOTE variants
        smote_methods = [
            ('regular', SMOTE(random_state=42, k_neighbors=1)),
            ('borderline', BorderlineSMOTE(random_state=42, k_neighbors=1)),
            ('adasyn', ADASYN(random_state=42, n_neighbors=1))
        ]
        
        for name, smote_method in smote_methods:
            try:
                print(f"Trying {name} SMOTE...")
                X_resampled, y_resampled = smote_method.fit_resample(X_rare_plus, y_rare_plus)
                print(f"Resampled with {name} SMOTE: {X_resampled.shape[0]} samples")
                
                # Count samples per class after resampling
                resampled_counts = pd.Series(y_resampled).value_counts()
                print(f"Class distribution after {name} SMOTE:")
                print(resampled_counts)
                
                # If successful, create a specialized classifier for rare classes
                rare_rf_params = {
                    'n_estimators': 500,  # More estimators for better rare class learning
                    'max_depth': 15,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,  # Less regularization to focus on rare patterns
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'n_jobs': -1,
                    'random_state': 42
                }
                
                rare_rf = RandomForestClassifier(**rare_rf_params)
                rare_rf.fit(X_resampled, y_resampled)
                
                # Add to our classifiers
                classifiers.append((f'rare_{name}_rf', Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', rare_rf)
                ])))
                
                break  # If we succeed with one SMOTE method, that's enough
                
            except Exception as e:
                print(f"Error applying {name} SMOTE: {e}")
    
    # Create the voting ensemble
    voting_classifier = VotingClassifier(
        estimators=classifiers,
        voting='soft',  # Use probability outputs
        n_jobs=-1
    )
    
    # Fit the ensemble
    print("\nTraining the ensemble model...")
    start_time = time.time()
    voting_classifier.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Ensemble training completed in {elapsed_time:.2f} seconds")
    
    # Predict with the ensemble
    y_pred = voting_classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nEnsemble model accuracy: {accuracy:.4f}")
    
    # Analyze how well we're doing with rare vs. common classes
    rare_indices_test = [i for i, label in enumerate(y_test) if label in rare_classes]
    common_indices_test = [i for i, label in enumerate(y_test) if label in common_classes]
    
    if rare_indices_test:
        rare_accuracy = accuracy_score([y_test[i] for i in rare_indices_test], 
                                      [y_pred[i] for i in rare_indices_test])
        print(f"Rare classes accuracy: {rare_accuracy:.4f}")
    
    common_accuracy = accuracy_score([y_test[i] for i in common_indices_test], 
                                   [y_pred[i] for i in common_indices_test])
    print(f"Common classes accuracy: {common_accuracy:.4f}")
    
    # Save model results
    class_names = sorted(set(y_train) | set(y_test))
    params = {
        'ensemble_type': 'VotingClassifier',
        'voting': 'soft',
        'base_estimators': ', '.join([name for name, _ in classifiers]),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'rare_threshold': rare_threshold,
        'rare_classes': ', '.join(rare_classes),
        'common_classes': ', '.join(common_classes)
    }
    
    save_model_results(voting_classifier, X_test, y_test, y_pred, "ensemble", class_names, params, output_dir=model_output_dir)
    
    # Examine predictions for rare classes in more detail
    if rare_classes:
        print("\nDetailed predictions for rare classes:")
        for cls in rare_classes:
            cls_indices = [i for i, label in enumerate(y_test) if label == cls]
            if cls_indices:
                correct = sum(y_test[i] == y_pred[i] for i in cls_indices)
                print(f"  {cls}: {correct}/{len(cls_indices)} correct ({correct/len(cls_indices):.2%})")
                
                # Show what the misclassified samples were predicted as
                if correct < len(cls_indices):
                    misclassified = [(i, y_pred[i]) for i in cls_indices if y_test[i] != y_pred[i]]
                    print(f"    Misclassified as: {[pred for _, pred in misclassified]}")
    
    return voting_classifier, X_test, y_test, y_pred

if __name__ == "__main__":
    train_ensemble_model() 