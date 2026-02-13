"""
Penguin Species Classification Model Training Script
Trains a Random Forest Classifier on Palmer Penguins dataset
"""
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import seaborn as sns
import pandas as pd

def train_model():
    """
    Train Random Forest Classifier on Penguins dataset with evaluation
    """
    # Load dataset
    print("Loading Palmer Penguins dataset...")
    penguins = sns.load_dataset('penguins')
    
    # Display dataset info
    print(f"\nDataset info:")
    print(f"- Original samples: {len(penguins)}")
    print(f"- Species: {penguins['species'].unique()}")
    
    # Drop rows with missing values
    penguins_clean = penguins.dropna()
    print(f"- Clean samples (after removing NaN): {len(penguins_clean)}")
    
    # Select features for classification
    feature_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    X = penguins_clean[feature_columns].values
    y = penguins_clean['species'].values
    
    # Create species mapping
    species_names = sorted(penguins_clean['species'].unique())
    species_to_id = {name: idx for idx, name in enumerate(species_names)}
    y_encoded = [species_to_id[species] for species in y]
    
    print(f"- Number of features: {len(feature_columns)}")
    print(f"- Features: {feature_columns}")
    print(f"- Species mapping: {species_to_id}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "model_type": "RandomForestClassifier",
        "dataset": "Palmer Penguins",
        "n_estimators": 100,
        "training_date": datetime.now().isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(feature_columns),
        "feature_names": feature_columns,
        "species_mapping": species_to_id,
        "n_classes": len(species_names)
    }
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=species_names))
    
    # Save model
    model_dir = '../model'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'penguin_rf_model.pkl')
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'model_metrics.json')
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✓ Model training completed successfully!")
    print(f"\nYou can now start the API with: uvicorn main:app --reload")
    return model, metrics

if __name__ == "__main__":
    train_model()