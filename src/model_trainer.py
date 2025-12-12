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

# --- 1. Model Definition (UPDATED ARCHITECTURE) ---
def build_lstm_model(input_shape):
    """
    Defines a deeper LSTM architecture with increased units and reduced dropout
    based on successful experiments to improve feature learning ability.
    """
    # Force float32 for consistency
    tf.keras.backend.set_floatx('float32') 
    
    model = Sequential([
        # LSTM layer 1: 100 units, return sequences for the next LSTM layer
        LSTM(100, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.1), 

        # LSTM layer 2: 100 units, returns a single output vector
        LSTM(100, return_sequences=False, activation='tanh'),
        Dropout(0.1),

        # Dense layer: Increased units to 50 for deeper feature processing
        Dense(units=50, activation='relu'),

        # Output layer
        Dense(units=1, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# --- 2. Sequence Generation Utility ---
def sequence_generator(df, features, target=None, window_length=50):
    """Generates 3D sequences (N_samples, WINDOW, N_features) for LSTM input."""
    sequences = []
    targets = []
    
    # RUL is available only in the training data
    if target: 
        # Train data: iterate over all engines to get all possible sequences
        for engine_id in df['Engine_No'].unique():
            engine_df = df[df['Engine_No'] == engine_id].reset_index(drop=True)
            data = engine_df[features].values
            
            # Start from the point where the first window can be created
            for i in range(window_length, len(engine_df) + 1, 1):
                sequences.append(data[i - window_length:i, :])
                targets.append(engine_df.loc[i - 1, target])
    else: 
        pass 
            
    if not sequences:
        return np.array([]), np.array([])
        
    return np.array(sequences), np.array(targets)


# --- 3. Evaluation Metric (NASA S-Score) ---
def calculate_s_score(y_true, y_pred):
    """Calculates the RUL S-Score (NASA challenge metric)."""
    difference = y_pred - y_true
    s_score = 0
    # Penalizes late predictions (d >= 0) more heavily (eta=10 vs eta=13)
    for d in difference:
        s_score += np.exp(d / 10.0) - 1 if d >= 0 else np.exp(-d / 13.0) - 1
    return s_score

# --- 4. Main Model Trainer Class ---
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.artifact_paths = config['artifact_paths']
        self.model_constants = config['model_constants']
        self.window_length = self.model_constants['window_length']
        self.rul_limit = self.model_constants['rul_limit']
        
        # --- FIX APPLIED HERE ---
        # The paths in config.yaml already contain 'artifacts', so we use them directly.
        
        # Define output paths (Model saving path)
        self.model_dir = self.artifact_paths['model_dir'] 
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load processed data (Training data path)
        processed_train_path = os.path.join(self.artifact_paths['processed_data_dir'], 'processed_train.csv')
        self.train_df = pd.read_csv(processed_train_path)
        
        logger.info(f"Loaded processed data from {processed_train_path}. Shape: {self.train_df.shape}")

    def train_and_evaluate(self):
        
        # Get the feature columns (all columns except Engine_No, Cycle, RUL)
        feature_cols = [
            col for col in self.train_df.columns 
            if col not in ['Engine_No', 'Cycle', 'RUL']
        ]
        
        # Generate sequences
        X_train_raw, y_train_raw = sequence_generator(
            self.train_df, features=feature_cols, target='RUL', window_length=self.window_length
        )
        
        # Shuffle the training data
        index = np.random.permutation(len(X_train_raw))
        X_train, y_train = X_train_raw[index], y_train_raw[index]
        
        logger.info(f"Training data sequences created. Shape: {X_train.shape}, Target shape: {y_train.shape}")
        
        # Build the model with the new architecture
        model = build_lstm_model(input_shape=(self.window_length, len(feature_cols)))
        
        # Define Callbacks
        patience = self.model_constants.get('patience', 10)
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True
        )
        model_save_path = os.path.join(self.model_dir, 'best_rul_model.keras')
        model_checkpoint = ModelCheckpoint(
            model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0
        )
        
        # Training
        EPOCHS = self.model_constants.get('epochs', 100)
        BATCH_SIZE = self.model_constants.get('batch_size', 128)
        
        logger.info(f"Starting training with EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
        # Log the model summary for tracking
        model.summary(print_fn=lambda x: logger.info(x))

        history = model.fit(
            X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=2,
            callbacks=[early_stopping, model_checkpoint], shuffle=True
        )
        
        logger.info("Training finished. Evaluating best model...")
        
        # --- Evaluation (on the validation set metrics after restoration) ---
        # Note: argmin is used to find the index (epoch - 1) of the minimum loss
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = history.history['val_loss'][best_epoch - 1]
        best_val_mae = history.history['val_mae'][best_epoch - 1]
        
        metrics = {
            'best_val_loss': float(best_val_loss),
            'best_val_mae': float(best_val_mae),
            'best_epoch': int(best_epoch),
            'features_used': len(feature_cols),
            'model_architecture': 'Deeper_LSTM_2x100_D0.1_RollingMean' # Updated tag
        }
        
        # Save metrics to YAML
        metrics_path = os.path.join(self.model_dir, 'metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f)
            
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Final Model saved at: {model_save_path}")
        
# --- Main Execution ---
if __name__ == "__main__":
    
    logger.info("--- Starting Model Training Pipeline ---")
    config = load_config()
    
    # Add optional keys if not in config
    if 'epochs' not in config['model_constants']: config['model_constants']['epochs'] = 100
    if 'batch_size' not in config['model_constants']: config['model_constants']['batch_size'] = 128
    
    trainer = ModelTrainer(config)
    
    trainer.train_and_evaluate()
    
    logger.info("Model Training pipeline step finished successfully.")