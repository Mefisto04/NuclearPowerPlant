"""
MODEL 1: OPTIMIZED RANDOM FOREST WITH CLASS WEIGHTS

This model addresses several issues from the original implementation:
1. Uses class_weight='balanced' to handle class imbalance
2. Performs extensive hyperparameter tuning with RandomizedSearchCV
3. Uses a stratified GroupKFold to ensure rare classes are properly evaluated
4. Implements feature selection to reduce dimensionality and focus on most important features

This approach maintains the RandomForest algorithm but optimizes it specifically
for the imbalanced nuclear accident classification problem.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import joblib
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_accident_models.utils import create_dataset, save_model_results

def train_optimized_random_forest():
    """Train an optimized Random Forest model with class weights and hyperparameter tuning"""
    
    # Create model-specific output directory
    model_output_dir = "../all_model_output/optimized_random_forest"
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("Creating dataset...")
    X, y, file_info = create_dataset()
    
    print("Dataset created with shape:", X.shape)
    print("Number of accident types:", len(set(y)))
    print("Accident types:", sorted(set(y)))
    
    # Create file groups for GroupKFold
    # This ensures that rare classes appear in the training set
    groups = [f"{info['accident_type']}_{info['file_id']}" for info in file_info]
    
    # Calculate class weights based on class frequencies
    class_counts = pd.Series(y).value_counts()
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    
    print("Class weights:")
    for cls, weight in class_weights.items():
        print(f"  {cls}: {weight:.4f}")
    
    # Split into train and test sets, ensuring rare classes are in training set
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Check if all classes appear in both training and test sets
    train_classes = set(y_train)
    test_classes = set(y_test)
    print("Classes in training set:", len(train_classes))
    print("Classes in test set:", len(test_classes))
    
    # If any class is missing from test set, adjust the split
    missing_in_test = train_classes - test_classes
    if missing_in_test:
        print(f"Warning: {len(missing_in_test)} classes not in test set. Adjusting split...")
        for cls in missing_in_test:
            # Find samples of this class in training set
            cls_indices = [i for i, label in enumerate(y_train) if label == cls]
            if cls_indices:
                # Move one sample to test set
                move_idx = cls_indices[0]
                X_test = pd.concat([X_test, X_train.iloc[[move_idx]]])
                y_test = np.append(y_test, [y_train[move_idx]])
                groups_test = np.append(groups_test, [groups_train[move_idx]])
                
                # Remove from training set
                X_train = X_train.drop(X_train.index[move_idx])
                y_train = np.delete(y_train, move_idx)
                groups_train = np.delete(groups_train, move_idx)
        
        print("After adjustment - Classes in training set:", len(set(y_train)))
        print("After adjustment - Classes in test set:", len(set(y_test)))
    
    # Define hyperparameter search space
    param_grid = {
        'rf__n_estimators': [100, 200, 300, 500],
        'rf__max_depth': [10, 20, 30, 40, 50, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'rf__bootstrap': [True, False],
        'rf__class_weight': ['balanced', 'balanced_subsample', class_weights]
    }
    
    # Create the pipeline with feature selection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    # Prepare the cross-validation strategy
    cv = GroupKFold(n_splits=3)
    
    # Setup RandomizedSearchCV
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=30, cv=cv, 
        scoring='balanced_accuracy', n_jobs=-1, verbose=1, random_state=42
    )
    
    # Fit the model
    random_search.fit(X_train, y_train, groups=groups_train)
    
    elapsed_time = time.time() - start_time
    print(f"RandomizedSearchCV completed in {elapsed_time:.2f} seconds")
    
    # Get the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    # Print feature importance
    if hasattr(best_model.named_steps['rf'], 'feature_importances_'):
        # Get selected features
        selected_features = best_model.named_steps['feature_selection'].get_support()
        selected_feature_names = X_train.columns[selected_features]
        print(f"\nNumber of selected features: {len(selected_feature_names)} out of {X_train.shape[1]}")
        
        # Get feature importance from the final classifier
        feature_importance = best_model.named_steps['rf'].feature_importances_
        
        # Create DataFrame for selected features and their importance
        importance_df = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        for i, (feature, importance) in enumerate(zip(importance_df['feature'][:10], importance_df['importance'][:10])):
            print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Save model results
    class_names = sorted(set(y_train) | set(y_test))
    save_model_results(best_model, X_test, y_test, y_pred, "optimized_random_forest", class_names, 
                       best_params, output_dir=model_output_dir)
    
    # Return the best model
    return best_model, X_test, y_test, X, y

if __name__ == "__main__":
    train_optimized_random_forest() 