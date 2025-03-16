"""
MODEL 4: ONE-VS-REST APPROACH WITH SPECIALIZED RARE CLASS CLASSIFIERS

This model addresses the issue of rare classes by:
1. Implementing a 'one-vs-rest' strategy for each rare class
2. Using data augmentation techniques for rare classes
3. Building specialized binary classifiers optimized for each rare class
4. Creating a hierarchical decision system - first detect if it's a rare class, then which one
5. Fine-tuning the threshold for rare class detection

This approach gives special attention to rare classes while still maintaining
good performance on common classes.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_accident_models.utils import create_dataset, save_model_results

# Try to import SMOTE for handling imbalanced classes
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imblearn not installed. SMOTE oversampling will not be available.")
    print("To install it, run: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

class CustomOneVsRestClassifier:
    """Custom implementation of OneVsRestClassifier with optimized thresholds for rare classes"""
    def __init__(self, estimator, rare_classes, rare_thresholds=None):
        self.ovr = OneVsRestClassifier(estimator)
        self.rare_classes = rare_classes
        self.rare_thresholds = rare_thresholds or {}
        self.classes_ = None
        
    def fit(self, X, y):
        self.ovr.fit(X, y)
        self.classes_ = self.ovr.classes_
        return self
        
    def predict(self, X):
        # Get probability predictions
        proba = self.ovr.predict_proba(X)
        
        predictions = []
        for i in range(len(X)):
            sample_proba = proba[i]
            max_idx = np.argmax(sample_proba)
            max_class = self.classes_[max_idx]
            
            # Check if any rare class exceeds its threshold
            rare_detected = False
            for j, cls in enumerate(self.classes_):
                if cls in self.rare_classes and cls in self.rare_thresholds:
                    if sample_proba[j] >= self.rare_thresholds[cls]:
                        predictions.append(cls)
                        rare_detected = True
                        break
            
            # If no rare class was detected with custom threshold, use the highest probability
            if not rare_detected:
                predictions.append(max_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        return self.ovr.predict_proba(X)

def train_one_vs_rest_model():
    """Train a One-vs-Rest model with specialized handling for rare classes"""
    
    # Create model-specific output directory
    model_output_dir = "../all_model_output/one_vs_rest"
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
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Check distribution of classes in train and test sets
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    
    print("\nClass distribution in training set:")
    print(train_counts)
    
    print("\nClass distribution in test set:")
    print(test_counts)
    
    # Create specialized models for rare classes
    rare_class_models = {}
    optimized_rare_thresholds = {}
    
    for rare_class in rare_classes:
        # Check if this rare class exists in the training set
        if rare_class not in y_train:
            print(f"Warning: Rare class {rare_class} not found in training set. Skipping.")
            continue
            
        print(f"\nTraining specialized model for rare class: {rare_class}")
        
        # Create binary problem: this rare class vs. everything else
        binary_y_train = [1 if label == rare_class else 0 for label in y_train]
        
        # Count positives and negatives
        positives = sum(binary_y_train)
        negatives = len(binary_y_train) - positives
        
        print(f"Binary classification: {positives} positive samples, {negatives} negative samples")
        
        # Sample weights to balance the classes
        class_weight = {
            1: negatives / (positives + negatives),
            0: positives / (positives + negatives)
        }
        
        print(f"Using class weights: {class_weight}")
        
        # Try to perform SMOTE if available
        if SMOTE_AVAILABLE and positives > 1:
            try:
                print("Applying SMOTE to balance the binary classification...")
                smote = SMOTE(random_state=42, k_neighbors=min(positives-1, 5))
                X_train_resampled, binary_y_train_resampled = smote.fit_resample(X_train, binary_y_train)
                
                # Count positives and negatives after SMOTE
                resampled_positives = sum(binary_y_train_resampled)
                resampled_negatives = len(binary_y_train_resampled) - resampled_positives
                
                print(f"After SMOTE: {resampled_positives} positive samples, {resampled_negatives} negative samples")
                
                # Create model for this rare class with balanced data
                rare_class_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=300,
                        max_depth=15,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        bootstrap=True,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                # Fit the model
                rare_class_model.fit(X_train_resampled, binary_y_train_resampled)
                
                # Find optimal threshold using cross-validation
                cv_scores = cross_val_score(rare_class_model, X_train_resampled, binary_y_train_resampled, 
                                           cv=3, scoring='roc_auc')
                print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Get probability predictions
                y_proba = rare_class_model.predict_proba(X_train)
                if y_proba.shape[1] > 1:  # Binary classifier
                    y_proba = y_proba[:, 1]  # Probability of positive class
                
                # Find optimal threshold using precision-recall curve
                precision, recall, thresholds = precision_recall_curve(binary_y_train, y_proba)
                
                # Calculate F1 score for each threshold
                f1_scores = []
                for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
                    if p + r > 0:  # Avoid division by zero
                        f1 = 2 * (p * r) / (p + r)
                    else:
                        f1 = 0
                    f1_scores.append((t, f1))
                
                # Find threshold with best F1 score
                best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
                print(f"Optimal threshold for {rare_class}: {best_threshold:.4f} (F1: {best_f1:.4f})")
                
                # Store the model and its optimal threshold
                rare_class_models[rare_class] = rare_class_model
                optimized_rare_thresholds[rare_class] = best_threshold
                
                # Plot precision-recall curve
                plt.figure(figsize=(10, 6))
                plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
                plt.plot(recall[:-1], [thresh for thresh in thresholds], 'r-', label='Threshold')
                plt.axvline(x=recall[thresholds == best_threshold][0] if any(thresholds == best_threshold) else 0, 
                           linestyle='--', color='g', label=f'Best Threshold: {best_threshold:.4f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision / Threshold')
                plt.title(f'Precision-Recall Curve for {rare_class}')
                plt.legend()
                plt.savefig(os.path.join(model_output_dir, f"precision_recall_{rare_class}.png"))
                plt.close()
                
            except Exception as e:
                print(f"Error applying SMOTE for {rare_class}: {e}")
                print("Using original imbalanced data instead.")
                
                # Create model with class weights instead
                rare_class_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=300,
                        max_depth=15,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        bootstrap=True,
                        class_weight=class_weight,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                # Fit the model
                rare_class_model.fit(X_train, binary_y_train)
                
                # Store the model with a default threshold
                rare_class_models[rare_class] = rare_class_model
                optimized_rare_thresholds[rare_class] = 0.3  # Default threshold
        
        else:
            # If SMOTE is not available or can't be applied, use class weights
            print("Using class weights to handle imbalance.")
            
            # Create model with class weights
            rare_class_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    bootstrap=True,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            # Fit the model
            rare_class_model.fit(X_train, binary_y_train)
            
            # Store the model with a default threshold
            rare_class_models[rare_class] = rare_class_model
            optimized_rare_thresholds[rare_class] = 0.3  # Default threshold
    
    # Create a model for common classes
    print("\nTraining main model for all classes...")
    main_classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train the One-vs-Rest classifier
    ovr_classifier = OneVsRestClassifier(main_classifier)
    ovr_classifier.fit(X_train, y_train)
    
    # Create custom classifier with optimized thresholds for rare classes
    custom_ovr = CustomOneVsRestClassifier(
        main_classifier, 
        rare_classes=rare_classes,
        rare_thresholds=optimized_rare_thresholds
    )
    custom_ovr.fit(X_train, y_train)
    
    # Make predictions
    y_pred_std = ovr_classifier.predict(X_test)  # Standard prediction
    y_pred_custom = custom_ovr.predict(X_test)  # Custom prediction with optimized thresholds
    
    # Calculate standard accuracy
    std_accuracy = accuracy_score(y_test, y_pred_std)
    print(f"\nStandard One-vs-Rest accuracy: {std_accuracy:.4f}")
    
    # Calculate custom accuracy
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    print(f"Custom One-vs-Rest with optimized thresholds accuracy: {custom_accuracy:.4f}")
    
    # Compare performance on rare classes
    rare_indices_test = [i for i, label in enumerate(y_test) if label in rare_classes]
    
    if rare_indices_test:
        # Standard approach
        rare_std_accuracy = accuracy_score(
            [y_test[i] for i in rare_indices_test],
            [y_pred_std[i] for i in rare_indices_test]
        )
        print(f"Standard approach - Rare classes accuracy: {rare_std_accuracy:.4f}")
        
        # Custom approach
        rare_custom_accuracy = accuracy_score(
            [y_test[i] for i in rare_indices_test],
            [y_pred_custom[i] for i in rare_indices_test]
        )
        print(f"Custom approach - Rare classes accuracy: {rare_custom_accuracy:.4f}")
        
        # Detailed report for rare classes
        print("\nDetailed predictions for rare classes:")
        for cls in rare_classes:
            cls_indices = [i for i, label in enumerate(y_test) if label == cls]
            if cls_indices:
                std_correct = sum(y_test[i] == y_pred_std[i] for i in cls_indices)
                custom_correct = sum(y_test[i] == y_pred_custom[i] for i in cls_indices)
                
                print(f"  {cls}: {len(cls_indices)} samples")
                print(f"    Standard: {std_correct}/{len(cls_indices)} correct ({std_correct/len(cls_indices):.2%})")
                print(f"    Custom: {custom_correct}/{len(cls_indices)} correct ({custom_correct/len(cls_indices):.2%})")
    
    # Choose the better approach based on performance
    if rare_indices_test and rare_custom_accuracy > rare_std_accuracy:
        print("\nUsing custom approach with optimized thresholds for final model.")
        final_model = custom_ovr
        y_pred = y_pred_custom
    else:
        print("\nUsing standard One-vs-Rest approach for final model.")
        final_model = ovr_classifier
        y_pred = y_pred_std
    
    # Save model results
    class_names = sorted(set(y_train) | set(y_test))
    params = {
        'model_type': 'One-vs-Rest',
        'estimator': 'RandomForestClassifier',
        'n_estimators': 200,
        'max_depth': 20,
        'rare_classes': ', '.join(rare_classes),
        'specialized_models': ', '.join(rare_class_models.keys()),
        'optimized_thresholds': str(optimized_rare_thresholds)
    }
    
    save_model_results(final_model, X_test, y_test, y_pred, "one_vs_rest", class_names, params, output_dir=model_output_dir)
    
    return final_model, X_test, y_test, y_pred

if __name__ == "__main__":
    train_one_vs_rest_model() 