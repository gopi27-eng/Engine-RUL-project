# src/model_trainer.py

import os
import sys
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Assuming utility files exist
from utils.logger import logger 
# You might need another file to load the config, or use a simplified helper here

# --- Configuration Loader (Utility function) ---
def load_config(config_path='config/config.yaml'):
    """Loads configuration from the specified YAML file."""
    try:
        config_path = os.path.join(os.getcwd(), config_path)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

# --- 1. Sequence Generator (Core Logic from Notebook) ---

def sequence_generator(df: pd.DataFrame, features: list, target: str, window_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping sequences (windows) of time-series data for training.
    """
    sequences = []
    targets = []
    
    # Iterate over each engine
    for engine_id in df['Engine_No'].unique():
        engine_df = df[df['Engine_No'] == engine_id].reset_index(drop=True)
        data = engine_df[features].values
        
        # Create sequences for training (all cycles up to the last one)
        for i in range(window_length, len(engine_df) + 1, 1):
            sequences.append(data[i - window_length:i, :])
            # The target RUL is taken from the last cycle in the window (index i-1)
            targets.append(engine_df.loc[i - 1, target])
            
    return np.array(sequences), np.array(targets)


# --- 2. Optimized Model Definition ---

def build_lstm_model(input_shape: tuple) -> Sequential:
    """Defines the optimized Deep LSTM architecture (100, 100 units)."""
    model = Sequential([
        # LSTM layer 1: 100 units, returns sequences for the next LSTM layer
        LSTM(100, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.1), 

        # LSTM layer 2: 100 units, returns only the last output sequence (for the Dense layer)
        LSTM(100, return_sequences=False, activation='tanh'),
        Dropout(0.1), 

        # Dense layer: 50 units (Increased capacity)
        Dense(units=50, activation='relu'),

        # Output layer (Linear activation for regression)
        Dense(units=1, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    logger.info("Deep LSTM model architecture defined and compiled.")
    return model

# --- 3. Evaluation Metric (NASA S-Score) ---

def calculate_s_score(y_true, y_pred):
    """Calculates the RUL S-Score (NASA challenge metric)."""
    difference = y_pred - y_true
    s_score = 0
    for d in difference:
        # Penalizes late predictions (d >= 0) more severely (eta=10)
        s_score += np.exp(d / 10.0) - 1 if d >= 0 else np.exp(-d / 13.0) - 1
    return s_score


# --- 4. Main Model Trainer Class ---
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.artifact_paths = config['artifact_paths']
        self.model_constants = config['model_constants']
        
        # Create necessary directories
        os.makedirs(self.artifact_paths['model_dir'], exist_ok=True)
        logger.info(f"Created model artifact directory: {self.artifact_paths['model_dir']}")

    def initiate_model_training(self, processed_train_path: str, processed_test_path: str, true_rul_path: str):
        
        logger.info("Starting Model Training component.")
        
        # --- 4.1 Load Processed Data ---
        train_df_scaled = pd.read_csv(processed_train_path)
        test_df_scaled = pd.read_csv(processed_test_path)
        true_rul_df = pd.read_csv(true_rul_path)
        
        # Define features used in preprocessing
        feature_cols = [col for col in train_df_scaled.columns if col not in ['Engine_No', 'Cycle', 'RUL']]
        
        # --- 4.2 Sequence Generation (Windowing) ---
        WINDOW_LENGTH = self.model_constants['window_length']
        
        logger.info(f"Generating sequences with window length: {WINDOW_LENGTH}")
        
        # Training sequences
        X_train_seq, y_train_target = sequence_generator(
            train_df_scaled, features=feature_cols, target='RUL', window_length=WINDOW_LENGTH
        )
        
        # Test sequences (Final window for each engine)
        # Note: We must regenerate this logic here as it was slightly different for the test set
        X_test_seq = []
        for engine_id in test_df_scaled['Engine_No'].unique():
            engine_df = test_df_scaled[test_df_scaled['Engine_No'] == engine_id]
            data = engine_df[feature_cols].values
            
            # Take only the last 'WINDOW_LENGTH' cycles, padding if necessary
            if len(data) >= WINDOW_LENGTH:
                X_test_seq.append(data[-WINDOW_LENGTH:])
            else:
                # Padding logic if engine test run is shorter than window (rare for FD001 but robust)
                padding_needed = WINDOW_LENGTH - len(data)
                padding = np.zeros((padding_needed, data.shape[1]))
                X_test_seq.append(np.vstack([padding, data]))
        
        X_test_seq = np.array(X_test_seq)
        y_true = true_rul_df['RUL'].values

        logger.info(f"Training sequences shape: {X_train_seq.shape}")

        # --- 4.3 Model Setup and Training ---
        
        # Shuffle training data
        index = np.random.permutation(len(X_train_seq))
        X_train_seq, y_train_target = X_train_seq[index], y_train_target[index]

        N_FEATURES = len(feature_cols)
        lstm_model = build_lstm_model(input_shape=(WINDOW_LENGTH, N_FEATURES))
        
        # Callbacks
        model_path = os.path.join(self.artifact_paths['model_dir'], 'best_rul_model.keras')
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.model_constants['patience'], verbose=1, mode='min', restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0
        )
        
        logger.info(f"Starting training for max {self.model_constants['epochs']} epochs.")
        
        history = lstm_model.fit(
            X_train_seq, y_train_target, 
            epochs=self.model_constants['epochs'], 
            batch_size=self.model_constants['batch_size'], 
            validation_split=0.1, 
            verbose=2,
            callbacks=[early_stopping, model_checkpoint], 
            shuffle=True
        )
        
        # --- 4.4 Prediction and Evaluation ---
        
        logger.info("Training finished. Making predictions on the test set.")
        y_pred = lstm_model.predict(X_test_seq).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        s_score = calculate_s_score(y_true, y_pred)
        
        logger.info(f"--- FINAL MODEL METRICS ---")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"RUL S-Score (NASA Metric): {s_score:.4f}")
        
        # --- 4.5 Save Evaluation Metrics (as an artifact) ---
        metrics = {'rmse': rmse, 's_score': s_score}
        metrics_path = os.path.join(self.artifact_paths['model_dir'], 'metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_path}")

        return model_path, metrics_path


# --- Example Execution ---
if __name__ == "__main__":
    
    # NOTE: This requires Data Ingestion and Preprocessing to have successfully run locally 
    # and saved files into artifacts/processed_data.
    
    config = load_config()
    
    processed_dir = config['artifact_paths']['processed_data_dir']
    
    # Define paths to the processed artifacts
    processed_train = os.path.join(processed_dir, 'processed_train.csv')
    processed_test = os.path.join(processed_dir, 'processed_test.csv')
    true_rul_path = os.path.join(processed_dir, 'true_rul.csv')
    
    if not all(os.path.exists(f) for f in [processed_train, processed_test, true_rul_path]):
        logger.error("Processed data artifacts not found. Please run data_preprocessing.py first.")
        sys.exit(1)

    trainer = ModelTrainer(config)
    final_model_path, final_metrics_path = trainer.initiate_model_training(
        processed_train, processed_test, true_rul_path
    )
    
    logger.info("Model Training pipeline step finished successfully.")
    logger.info(f"Final Model saved at: {final_model_path}")