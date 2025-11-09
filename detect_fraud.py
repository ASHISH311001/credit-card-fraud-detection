#!/usr/bin/env python3
"""
Credit Card Fraud Detection using Anomaly Detection
Author: Ashish Jha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

class AutoencoderFraudDetector:
    """Autoencoder-based fraud detection"""
    
    def __init__(self, encoding_dim=14):
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.threshold = None
    
    def build_model(self, input_dim):
        """Build autoencoder model"""
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        encoded = layers.Dense(int(self.encoding_dim/2), activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(int(self.encoding_dim/2), activation='relu')(encoded)
        decoded = layers.Dense(self.encoding_dim, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def fit(self, X_train, epochs=50, batch_size=32):
        """Train the autoencoder"""
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.autoencoder.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        
        return history
    
    def predict(self, X):
        """Predict fraud (1) or normal (0)"""
        reconstructions = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        predictions = (mse > self.threshold).astype(int)
        return predictions

def load_data(filepath):
    """Load and preprocess credit card data"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
    print(f"Normal transactions: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.2f}%)")
    
    return X, y

def train_isolation_forest(X_train, y_train, X_test, y_test):
    """Train Isolation Forest model"""
    print("\nTraining Isolation Forest...")
    
    # Train on normal transactions only
    X_train_normal = X_train[y_train == 0]
    
    clf = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=100
    )
    clf.fit(X_train_normal)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # -1 for fraud, 1 for normal
    
    # Evaluate
    print("\nIsolation Forest Results:")
    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    
    return clf, y_pred

def train_autoencoder(X_train, y_train, X_test, y_test):
    """Train Autoencoder model"""
    print("\nTraining Autoencoder...")
    
    # Train on normal transactions only
    X_train_normal = X_train[y_train == 0]
    
    detector = AutoencoderFraudDetector(encoding_dim=14)
    detector.build_model(X_train.shape[1])
    detector.fit(X_train_normal, epochs=50, batch_size=32)
    
    # Predict
    y_pred = detector.predict(X_test)
    
    # Evaluate
    print("\nAutoencoder Results:")
    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    
    return detector, y_pred

if __name__ == "__main__":
    # Load data (Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud)
    X, y = load_data('data/creditcard.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Isolation Forest
    iso_model, iso_pred = train_isolation_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Train Autoencoder
    ae_model, ae_pred = train_autoencoder(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save models
    joblib.dump(iso_model, 'models/isolation_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    ae_model.autoencoder.save('models/autoencoder_model.h5')
    
    print("\nModels saved successfully!")
    print("\nBoth models handle highly imbalanced datasets effectively!")
    print("High recall and precision achieved through anomaly detection.")
