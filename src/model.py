"""
Machine Learning Model Module
=============================
Trains and applies land cover classification models.
Uses scikit-learn (FREE) — no paid ML platforms required.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            cohen_kappa_score)
import pickle
from pathlib import Path

class LandCoverClassifier:
    """
    Random Forest classifier for land cover classification.

    Classes:
    --------
    0: Water
    1: Forest
    2: Agriculture
    3: Urban
    4: Barren
    5: Grassland
    6: Wetland
    """

    CLASSES = ['Water', 'Forest', 'Agriculture', 'Urban', 
               'Barren', 'Grassland', 'Wetland']

    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.is_trained = False
        self.metrics = {}

    def prepare_features(self, rgb, nir, swir):
        """
        Extract features from satellite bands.

        Features:
        - Spectral bands (R, G, B, NIR, SWIR)
        - NDVI (vegetation)
        - NDWI (water)
        - Texture indices
        """
        # Flatten images
        h, w = rgb.shape[:2]
        n_pixels = h * w

        # Spectral features
        features = np.zeros((n_pixels, 8))
        features[:, 0] = rgb[:,:,0].flatten() / 255.0  # Red
        features[:, 1] = rgb[:,:,1].flatten() / 255.0  # Green
        features[:, 2] = rgb[:,:,2].flatten() / 255.0  # Blue
        features[:, 3] = nir.flatten() / 255.0          # NIR
        features[:, 4] = swir.flatten() / 255.0         # SWIR

        # Calculate indices
        red = features[:, 0]
        green = features[:, 1]
        nir_norm = features[:, 3]

        # NDVI
        ndvi = (nir_norm - red) / (nir_norm + red + 0.001)
        features[:, 5] = ndvi

        # NDWI
        ndwi = (green - nir_norm) / (green + nir_norm + 0.001)
        features[:, 6] = ndwi

        # Simple texture (standard deviation in 3x3 window)
        texture = np.zeros((h, w))
        for i in range(1, h-1):
            for j in range(1, w-1):
                window = nir[i-1:i+2, j-1:j+2]
                texture[i, j] = np.std(window)
        features[:, 7] = texture.flatten() / 255.0

        return features

    def generate_synthetic_labels(self, features):
        """
        Generate synthetic labels for demonstration.
        In production, you would use ground truth data.
        """
        np.random.seed(42)
        n_samples = features.shape[0]

        # Use NDVI and NDWI to create realistic labels
        ndvi = features[:, 5]
        ndwi = features[:, 6]

        labels = np.zeros(n_samples, dtype=int)

        # Water: high NDWI
        labels[ndwi > 0.3] = 0

        # Forest: high NDVI
        labels[(ndvi > 0.5) & (ndwi <= 0.3)] = 1

        # Agriculture: moderate NDVI
        labels[(ndvi > 0.3) & (ndvi <= 0.5) & (ndwi <= 0.3)] = 2

        # Urban: low NDVI, high SWIR
        labels[(ndvi < 0.2) & (features[:, 4] > 0.4)] = 3

        # Barren: very low NDVI
        labels[(ndvi < 0.1) & (labels == 0)] = 4

        # Grassland: low-moderate NDVI
        labels[(ndvi > 0.2) & (ndvi <= 0.3) & (labels == 0)] = 5

        # Wetland: moderate NDWI
        labels[(ndwi > 0.1) & (ndwi <= 0.3) & (labels == 0)] = 6

        return labels

    def train(self, rgb, nir, swir, labels=None):
        """
        Train the classification model.

        Parameters:
        -----------
        rgb : np.ndarray
            RGB image (H, W, 3)
        nir : np.ndarray
            NIR band (H, W)
        swir : np.ndarray
            SWIR band (H, W)
        labels : np.ndarray, optional
            Ground truth labels. If None, synthetic labels are generated.
        """
        print("Extracting features...")
        features = self.prepare_features(rgb, nir, swir)

        if labels is None:
            print("Generating synthetic training labels...")
            labels = self.generate_synthetic_labels(features)
        else:
            labels = labels.flatten()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=self.CLASSES,
                                                          zero_division=0)
        }

        print(f"\nAccuracy: {self.metrics['accuracy']:.3f}")
        print(f"Kappa: {self.metrics['kappa']:.3f}")

        return self.metrics

    def predict(self, rgb, nir, swir):
        """Predict land cover classes for new imagery."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")

        features = self.prepare_features(rgb, nir, swir)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        # Reshape to image dimensions
        h, w = rgb.shape[:2]
        predictions = predictions.reshape(h, w)
        probabilities = probabilities.reshape(h, w, -1)

        return predictions, probabilities

    def save_model(self, filepath="models/random_forest_model.pkl"):
        """Save trained model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="models/random_forest_model.pkl"):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self):
        """Return feature importance scores."""
        if not self.is_trained:
            return None

        feature_names = ['Red', 'Green', 'Blue', 'NIR', 'SWIR', 
                        'NDVI', 'NDWI', 'Texture']
        importance = self.model.feature_importances_

        return dict(zip(feature_names, importance))


if __name__ == "__main__":
    # Test the classifier
    from data_loader import SatelliteDataLoader

    print("=" * 50)
    print("GeoAI Land Cover Classifier - Test Run")
    print("=" * 50)

    # Load data
    loader = SatelliteDataLoader()
    data = loader.load_sample_data("amazon")

    # Train model
    classifier = LandCoverClassifier(n_estimators=50)
    metrics = classifier.train(data['rgb'], data['nir'], data['swir'])

    # Feature importance
    importance = classifier.get_feature_importance()
    print("\nFeature Importance:")
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {score:.3f}")

    # Predict
    predictions, probabilities = classifier.predict(data['rgb'], data['nir'], data['swir'])
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Unique classes: {np.unique(predictions)}")
