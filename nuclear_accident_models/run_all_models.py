"""
MAIN SCRIPT FOR RUNNING ALL NUCLEAR ACCIDENT CLASSIFICATION MODELS

This script runs all four model implementations and compares their performance:
1. Optimized Random Forest with class weights and feature selection
2. Neural Network with class weighting
3. Ensemble of multiple classifiers
4. One-vs-Rest approach with specialized rare class classifiers

For each model, it tracks:
- Overall accuracy
- Per-class accuracy, especially for rare classes
- Training time
- Feature importance (where applicable)

The script then determines which model performs best for the nuclear accident classification task.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all model training functions
try:
    from nuclear_accident_models.model1_optimized_random_forest import train_optimized_random_forest
    from nuclear_accident_models.model2_neural_network import train_neural_network
    from nuclear_accident_models.model3_ensemble import train_ensemble_model
    from nuclear_accident_models.model4_one_vs_rest import train_one_vs_rest_model
    
    from nuclear_accident_models.utils import create_dataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the right directory.")
    sys.exit(1)

def get_model_output_dir(base_dir, model_name):
    """Create and return a model-specific output directory"""
    # Convert model name to a valid directory name (lowercase, no spaces)
    dir_name = model_name.lower().replace(' ', '_')
    model_dir = os.path.join(base_dir, dir_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def run_all_models():
    """Run all models and compare their performance"""
    
    # Directory for saving results
    base_output_dir = "../all_model_output"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a common directory for comparative results
    comparative_dir = os.path.join(base_output_dir, "comparative_results")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Create empty results dictionary
    results = {
        'model_name': [],
        'accuracy': [],
        'training_time': [],
        'rare_class_accuracy': [],
        'common_class_accuracy': []
    }
    
    # Load the dataset once for class information
    print("Loading dataset to identify rare classes...")
    X, y, file_info = create_dataset()
    class_counts = pd.Series(y).value_counts()
    rare_threshold = 5  # Classes with <= 5 samples
    
    rare_classes = class_counts[class_counts <= rare_threshold].index.tolist()
    common_classes = class_counts[class_counts > rare_threshold].index.tolist()
    
    print(f"Identified {len(rare_classes)} rare classes: {rare_classes}")
    print(f"Identified {len(common_classes)} common classes")
    print("Dataset shape:", X.shape)
    
    # Dictionary to store detailed per-class accuracy for each model
    class_accuracy = {}
    
    # Function to calculate per-class accuracy
    def get_per_class_accuracy(y_true, y_pred):
        per_class = {}
        classes = sorted(set(y_true))
        
        for cls in classes:
            # Get indices for this class
            indices = [i for i, label in enumerate(y_true) if label == cls]
            if indices:
                correct = sum(y_true[i] == y_pred[i] for i in indices)
                per_class[cls] = correct / len(indices)
            else:
                per_class[cls] = 0
                
        return per_class
    
    # Dictionary to store test data and predictions for each model
    all_predictions = {}
    
    # Run each model and collect results
    models_to_run = [
        ('Optimized Random Forest', train_optimized_random_forest),
        ('Neural Network', train_neural_network),
        ('Ensemble', train_ensemble_model),
        ('One-vs-Rest', train_one_vs_rest_model)
    ]
    
    for model_name, train_func in models_to_run:
        print(f"\n{'-'*80}")
        print(f"RUNNING {model_name.upper()}")
        print(f"{'-'*80}")
        
        # Create model-specific output directory
        model_output_dir = get_model_output_dir(base_output_dir, model_name)
        print(f"Model output will be saved to: {model_output_dir}")
        
        # Track training time
        start_time = time.time()
        
        try:
            # Train the model
            model, X_test, y_test, y_pred = train_func()
            
            # If train function returns a predict function instead of predictions
            if callable(y_pred):
                # This is likely the Neural Network case, where we get a prediction function
                predict_func = y_pred
                y_pred = predict_func(X_test)['predicted_type']
            
            training_time = time.time() - start_time
            
            # Calculate overall accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate accuracy for rare and common classes
            rare_indices = [i for i, label in enumerate(y_test) if label in rare_classes]
            common_indices = [i for i, label in enumerate(y_test) if label in common_classes]
            
            rare_acc = 0
            if rare_indices:
                rare_acc = accuracy_score(
                    [y_test[i] for i in rare_indices],
                    [y_pred[i] for i in rare_indices]
                )
            
            common_acc = 0
            if common_indices:
                common_acc = accuracy_score(
                    [y_test[i] for i in common_indices],
                    [y_pred[i] for i in common_indices]
                )
            
            # Store results
            results['model_name'].append(model_name)
            results['accuracy'].append(accuracy)
            results['training_time'].append(training_time)
            results['rare_class_accuracy'].append(rare_acc)
            results['common_class_accuracy'].append(common_acc)
            
            # Calculate per-class accuracy
            class_accuracy[model_name] = get_per_class_accuracy(y_test, y_pred)
            
            # Store predictions
            all_predictions[model_name] = {
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"\n{model_name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Training time: {training_time:.2f} seconds")
            print(f"  Rare class accuracy: {rare_acc:.4f}")
            print(f"  Common class accuracy: {common_acc:.4f}")
            
            # Save model-specific classification report and confusion matrix
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(model_output_dir, f"classification_report_{model_name.lower().replace(' ', '_')}.csv"))
            
            # Create and save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=sorted(set(y_test)), 
                        yticklabels=sorted(set(y_test)))
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(model_output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"))
            plt.close()
            
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create a comparative results table
    results_df = pd.DataFrame(results)
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n")
    print("="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))
    
    # Save comparative results to CSV
    results_df.to_csv(os.path.join(comparative_dir, "model_comparison.csv"), index=False)
    
    # Create detailed per-class accuracy comparison
    all_classes = sorted(set(y))
    per_class_df = pd.DataFrame({cls: [class_accuracy.get(model, {}).get(cls, 0) 
                                      for model in results['model_name']] 
                               for cls in all_classes}, 
                              index=results['model_name'])
    
    # Highlight rare classes
    rare_class_df = per_class_df[rare_classes]
    common_class_df = per_class_df[common_classes]
    
    print("\nAccuracy for Rare Classes:")
    print(tabulate(rare_class_df, headers='keys', tablefmt='pretty'))
    
    print("\nAverage Accuracy for Rare Classes:")
    rare_avg = rare_class_df.mean(axis=1)
    print(tabulate(pd.DataFrame({'Average': rare_avg}), headers='keys', tablefmt='pretty'))
    
    # Save per-class accuracy to CSV
    per_class_df.to_csv(os.path.join(comparative_dir, "per_class_accuracy.csv"))
    
    # Plot comparative results
    plt.figure(figsize=(12, 6))
    
    # Overall accuracy
    plt.subplot(1, 2, 1)
    plt.bar(results['model_name'], results['accuracy'])
    plt.title('Overall Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, acc in enumerate(results['accuracy']):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
    
    # Rare vs Common class accuracy
    plt.subplot(1, 2, 2)
    x = np.arange(len(results['model_name']))
    width = 0.35
    
    plt.bar(x - width/2, results['rare_class_accuracy'], width, label='Rare Classes')
    plt.bar(x + width/2, results['common_class_accuracy'], width, label='Common Classes')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Rare vs Common Class Accuracy')
    plt.xticks(x, results['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparative_dir, "model_comparison.png"))
    
    # Create ensemble of the best models
    print("\nCreating a final ensemble from the best models...")
    
    # Get the top 2 models
    top_models = results_df.head(2)['model_name'].tolist()
    print(f"Using top models for final ensemble: {top_models}")
    
    # Create a folder for the final ensemble
    ensemble_dir = os.path.join(base_output_dir, "final_ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # Create a voting ensemble
    final_predictions = []
    
    for i in range(len(all_predictions[top_models[0]]['y_test'])):
        votes = {}
        for model in top_models:
            pred = all_predictions[model]['y_pred'][i]
            votes[pred] = votes.get(pred, 0) + 1
        
        # Get the prediction with the most votes
        final_predictions.append(max(votes.items(), key=lambda x: x[1])[0])
    
    # Calculate final accuracy
    y_test = all_predictions[top_models[0]]['y_test']
    final_accuracy = accuracy_score(y_test, final_predictions)
    
    print(f"Final ensemble accuracy: {final_accuracy:.4f}")
    
    # Calculate per-class accuracy for final ensemble
    final_per_class = get_per_class_accuracy(y_test, final_predictions)
    
    # Calculate rare class accuracy
    rare_indices = [i for i, label in enumerate(y_test) if label in rare_classes]
    rare_acc = 0
    if rare_indices:
        rare_acc = accuracy_score(
            [y_test[i] for i in rare_indices],
            [final_predictions[i] for i in rare_indices]
        )
        print(f"Final ensemble rare class accuracy: {rare_acc:.4f}")
    
    # Report classification results
    report = classification_report(y_test, final_predictions)
    print("\nFinal Ensemble Classification Report:")
    print(report)
    
    # Save the report
    with open(os.path.join(ensemble_dir, "final_ensemble_report.txt"), 'w') as f:
        f.write("Final Ensemble of Top Models\n")
        f.write(f"Models used: {', '.join(top_models)}\n\n")
        f.write(f"Overall Accuracy: {final_accuracy:.4f}\n")
        f.write(f"Rare Class Accuracy: {rare_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Create confusion matrix for final ensemble
    cm = confusion_matrix(y_test, final_predictions)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_test)), 
                yticklabels=sorted(set(y_test)))
    plt.title('Final Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(ensemble_dir, "final_ensemble_confusion_matrix.png"))
    
    print(f"\nAll models have been evaluated. Results saved to: {base_output_dir}")
    
    # Determine the best model based on rare class performance
    best_rare_model = results_df.sort_values('rare_class_accuracy', ascending=False).iloc[0]
    best_overall_model = results_df.iloc[0]  # Already sorted by overall accuracy
    
    print("\nBest model for rare classes:")
    print(f"  {best_rare_model['model_name']} (Rare Class Accuracy: {best_rare_model['rare_class_accuracy']:.4f})")
    
    print("\nBest model overall:")
    print(f"  {best_overall_model['model_name']} (Overall Accuracy: {best_overall_model['accuracy']:.4f})")
    
    if final_accuracy > best_overall_model['accuracy']:
        print("\nFinal ensemble outperforms the best individual model!")
        print(f"  Ensemble Accuracy: {final_accuracy:.4f} vs. Best Individual: {best_overall_model['accuracy']:.4f}")
    
    return results_df, all_predictions

if __name__ == "__main__":
    run_all_models() 