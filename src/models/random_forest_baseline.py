"""
Random Forest Baseline Model for Single-Cell RNA-seq Classification

This module implements a Random Forest classifier as a classical machine learning baseline
for cell type or disease state prediction from single-cell gene expression data.

Key Features:
- Fast, interpretable classification
- Feature importance ranking
- Baseline for comparison with deep learning models

Author: AI Assistant
Date: 2025
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestBaseline:
    """
    Random Forest classifier for single-cell RNA-seq data.
    Provides fit, predict, and evaluation methods.
    """
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        """
        Initialize the Random Forest model.
        Args:
            n_estimators: Number of trees
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        self.is_trained = False

    def fit(self, X, y):
        """
        Train the Random Forest model.
        Args:
            X: Feature matrix
            y: Labels
        """
        logger.info("Training Random Forest model...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Random Forest training complete.")

    def predict(self, X):
        """
        Predict labels for new data.
        Args:
            X: Feature matrix
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        Args:
            X: Feature matrix
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict_proba(X)

    def evaluate(self, X, y_true):
        """
        Evaluate the model on test data.
        Args:
            X: Feature matrix
            y_true: True labels
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Random Forest model...")
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception:
            roc_auc = 0.0
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        logger.info(f"Random Forest evaluation complete. Accuracy: {accuracy:.4f}")
        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'classification_report': class_report,
            'feature_importances': self.model.feature_importances_.tolist()
        }

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Random Forest Baseline model...")
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    n_classes = 3
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    # Create and train model
    rf = RandomForestBaseline()
    rf.fit(X, y)
    # Evaluate
    results = rf.evaluate(X, y)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info("Random Forest Baseline model ready for use!")
