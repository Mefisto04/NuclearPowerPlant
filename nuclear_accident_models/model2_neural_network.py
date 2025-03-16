"""
MODEL 2: DEEP NEURAL NETWORK CLASSIFIER WITH CLASS WEIGHTING

This model introduces a fundamentally different approach to the classification problem:
1. Uses a neural network architecture instead of traditional ML algorithms
2. Applies class weights to address the imbalanced dataset problem
3. Uses batch normalization and dropout to prevent overfitting
4. Implements early stopping to find optimal training duration
5. Applies PCA for dimensionality reduction before training

This approach offers potentially better feature learning capabilities than
tree-based models, which might help with handling the rare classes.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_accident_models.utils import create_dataset, save_model_results

# Create a class to wrap Keras model into a scikit-learn compatible estimator
class KerasClassifierWrapper:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.classes_ = self.label_encoder.classes_
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    def predict_proba(self, X):
        return self.model.predict(X)

def train_neural_network():
    """Train a neural network classifier with class weighting"""
    
    # Create model-specific output directory
    model_output_dir = "../all_model_output/neural_network"
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("Creating dataset...")
    X, y, file_info = create_dataset()
    
    print("Dataset created with shape:", X.shape)
    print("Number of accident types:", len(set(y)))
    print("Accident types:", sorted(set(y)))
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Encoded {num_classes} classes")
    
    # Calculate class weights for imbalanced dataset
    class_counts = pd.Series(y).value_counts()
    total_samples = len(y)
    class_weights = {
        label_encoder.transform([cls])[0]: total_samples / (len(class_counts) * count) 
        for cls, count in class_counts.items()
    }
    
    print("Class weights:")
    for encoded, weight in class_weights.items():
        print(f"  {label_encoder.inverse_transform([encoded])[0]}: {weight:.4f}")
    
    # Split into train and test sets
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, shuffle=True, stratify=None
    )
    
    # Get original labels for test set
    y_test = label_encoder.inverse_transform(y_test_encoded)
    
    # Check if all classes appear in training and test sets (for encoded labels)
    train_classes = set(y_train_encoded)
    test_classes = set(y_test_encoded)
    print(f"Classes in training set: {len(train_classes)} / {num_classes}")
    print(f"Classes in test set: {len(test_classes)} / {num_classes}")
    
    # Apply scaling and PCA for dimensionality reduction
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA while keeping 95% of the variance
    print("Applying PCA...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced dimensions from {X_train.shape[1]} to {X_train_pca.shape[1]}")
    
    # Convert to one-hot encoding for neural network
    y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Define the neural network architecture
    input_dim = X_train_pca.shape[1]
    
    # Set up deterministic behavior for reproducibility
    tf.random.set_seed(42)
    
    # Build the model with appropriate architecture
    model = Sequential([
        # Input layer with batch normalization
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with appropriate loss and metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Define callbacks for early stopping and model checkpoint
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Set up model checkpoint
    checkpoint_path = os.path.join(model_output_dir, "nn_model_checkpoint.h5")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model with class weights
    print("Training neural network...")
    history = model.fit(
        X_train_pca, y_train_onehot,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, "neural_network_training_history.png"))
    plt.close()
    
    # Make predictions on test set
    y_pred_onehot = model.predict(X_test_pca)
    y_pred_encoded = np.argmax(y_pred_onehot, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Create a scikit-learn compatible wrapper for the model
    wrapped_model = KerasClassifierWrapper(model, label_encoder)
    
    # Save model results
    class_names = label_encoder.classes_
    params = {
        'architecture': 'Sequential NN',
        'input_dim': input_dim,
        'hidden_layers': '256-128-64',
        'activation': 'relu',
        'dropout_rates': '0.4-0.3-0.2',
        'batch_normalization': 'Yes',
        'optimizer': 'Adam(lr=0.001)',
        'batch_size': 32,
        'epochs': len(history.history['loss']),
        'early_stopping_patience': 15,
        'pca_components': input_dim
    }
    
    save_model_results(wrapped_model, pd.DataFrame(X_test_pca), y_test, y_pred, 
                       "neural_network", class_names, params, output_dir=model_output_dir)
    
    # Save the model in Keras format
    model.save(os.path.join(model_output_dir, "neural_network.h5"))
    
    # Also save the preprocessors
    import joblib
    joblib.dump(scaler, os.path.join(model_output_dir, "neural_network_scaler.joblib"))
    joblib.dump(pca, os.path.join(model_output_dir, "neural_network_pca.joblib"))
    joblib.dump(label_encoder, os.path.join(model_output_dir, "neural_network_encoder.joblib"))
    
    # Create a prediction function for easy reuse
    def predict_with_neural_network(new_data):
        # Preprocess
        new_data_scaled = scaler.transform(new_data)
        new_data_pca = pca.transform(new_data_scaled)
        
        # Predict
        pred_onehot = model.predict(new_data_pca)
        pred_encoded = np.argmax(pred_onehot, axis=1)
        pred = label_encoder.inverse_transform(pred_encoded)
        
        # Get probabilities for top classes
        top_n = 3
        top_classes = []
        top_probs = []
        
        for i in range(len(pred_onehot)):
            indices = np.argsort(pred_onehot[i])[::-1][:top_n]
            top_classes.append([label_encoder.inverse_transform([idx])[0] for idx in indices])
            top_probs.append([pred_onehot[i][idx] for idx in indices])
        
        return {
            'predicted_type': pred,
            'top_n_types': top_classes,
            'top_n_probs': top_probs
        }
    
    # Save the prediction function
    joblib.dump(predict_with_neural_network, os.path.join(model_output_dir, "neural_network_predict_fn.joblib"))
    
    # Return the important components
    return wrapped_model, X_test, y_test, predict_with_neural_network

if __name__ == "__main__":
    train_neural_network() 