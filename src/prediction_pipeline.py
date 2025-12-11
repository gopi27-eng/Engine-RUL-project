# src/prediction_pipeline.py

import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

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

# --- Core Prediction Class ---
class RULPredictor:
    def __init__(self, config):
        self.config = config
        self.artifact_paths = config['artifact_paths']
        self.model_constants = config['model_constants']
        
        # Define the 17 essential features by removing dropped and ID columns
        all_cols = config['column_names']
        dropped_ids_cycles = config['dropped_columns'] + ['Engine_No', 'Cycle']
        self.feature_cols = [col for col in all_cols if col not in dropped_ids_cycles]
        
        # Define artifact paths
        self.scaler_path = os.path.join(self.artifact_paths['scaler_dir'], 'minmax_scaler.pkl')
        self.model_path = os.path.join(self.artifact_paths['model_dir'], 'best_rul_model.keras')

        # Load artifacts
        try:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("MinMaxScaler loaded successfully.")
            
            self.model = load_model(self.model_path)
            logger.info("Trained LSTM Model loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"FATAL: Required artifact not found. Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            sys.exit(1)

    def preprocess_engine_data(self, df_new_engine: pd.DataFrame) -> np.ndarray:
        """
        Applies necessary preprocessing (feature selection, scaling, windowing) 
        to a single engine's time-series data.
        """
        
        WINDOW_LENGTH = self.model_constants['window_length']
        
        # 1. Feature Selection
        # Ensure the input DataFrame contains only the features used for training
        df_new_engine = df_new_engine[self.feature_cols]
        
        # 2. Scaling (using the pre-fitted training scaler)
        data_scaled = self.scaler.transform(df_new_engine.values)
        
        # 3. Windowing (Get the last sequence)
        if data_scaled.shape[0] < WINDOW_LENGTH:
            logger.warning(f"Engine data ({data_scaled.shape[0]} cycles) is shorter than window ({WINDOW_LENGTH}). Padding with zeros.")
            padding_needed = WINDOW_LENGTH - data_scaled.shape[0]
            padding = np.zeros((padding_needed, data_scaled.shape[1]))
            final_sequence = np.vstack([padding, data_scaled])
        else:
            final_sequence = data_scaled[-WINDOW_LENGTH:]
            
        # Convert to 3D tensor for LSTM
        return np.expand_dims(final_sequence, axis=0) # Shape (1, WINDOW_LENGTH, N_FEATURES)

    def predict_rul(self, data_sequence: np.ndarray) -> float:
        """Generates RUL prediction from the preprocessed sequence."""
        
        logger.info("Generating prediction...")
        prediction = self.model.predict(data_sequence, verbose=0)[0][0]
        
        # Cap the prediction at the RUL_LIMIT
        capped_prediction = min(prediction, self.model_constants['rul_limit'])
        
        return float(capped_prediction)


# --- Example Execution ---
if __name__ == "__main__":
    
    logger.info("--- Starting Prediction Pipeline ---")
    
    config = load_config()
    
    # 1. Initialize the Predictor (loads model and scaler)
    predictor = RULPredictor(config)

    # 2. SIMULATE LIVE DATA for a single engine (Engine ID 100 from the test set)
    try:
        # Load raw test data from the local path defined in config.yaml
        test_df_raw = pd.read_csv(
            os.path.join(config['local_data_paths']['root_dir'], 'test_FD001.txt'), 
            sep=r"\s+", header=None, names=config['column_names']
        )
        
        # --- FIX: Use dropna to safely remove only the trailing NaN columns ---
        test_df_raw = test_df_raw.dropna(axis=1, how='all')
        
        target_engine_id = 100
        # Get all cycles for the specific engine ID
        live_engine_data = test_df_raw[test_df_raw['Engine_No'] == target_engine_id].drop(columns=['Engine_No'])
        
        if live_engine_data.empty:
            logger.error(f"Engine ID {target_engine_id} not found in test data for simulation.")
            sys.exit(1)

        logger.info(f"Simulating latest data point for Engine ID {target_engine_id} (Total cycles: {live_engine_data['Cycle'].max()})")
        
    except Exception as e:
        logger.error(f"Failed to load test data for simulation: {e}")
        sys.exit(1)

    # 3. Preprocess
    sequence = predictor.preprocess_engine_data(live_engine_data)

    # 4. Predict
    predicted_rul = predictor.predict_rul(sequence)
    
    # 5. Output Result
    logger.info(f"\n=======================================================")
    logger.info(f"| FINAL RUL PREDICTION for Engine ID {target_engine_id}: {predicted_rul:.2f} cycles |")
    logger.info(f"=======================================================")
    
    logger.info("Prediction Pipeline step finished successfully.")