# api/prediction_service.py
import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

class RULPredictionService:
    def __init__(self, model_path, scaler_path, window_length=50):
        # Define paths relative to the project root (assuming API runs from a subdirectory or handles paths correctly)
        self.model_path = os.path.join(os.getcwd(), model_path)
        self.scaler_path = os.path.join(os.getcwd(), scaler_path)
        self.window_length = window_length
        self.model = None
        self.scaler = None
        self.feature_cols = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the trained Keras model and MinMaxScaler."""
        try:
            # 1. Load Model
            self.model = load_model(self.model_path)
            
            # 2. Load Scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # 3. Derive feature names from the scaler's input feature set 
            # (Note: In a real MLOps system, feature names should be saved explicitly)
            # For simplicity, we assume the original 32 features are loaded correctly
            # We must load a sample of the processed data to get the feature order
            
            processed_train_path = os.path.join(os.getcwd(), 'artifacts/processed_data/processed_train.csv')
            temp_df = pd.read_csv(processed_train_path)
            
            self.feature_cols = [
                col for col in temp_df.columns 
                if col not in ['Engine_No', 'Cycle', 'RUL']
            ]
            
            print(f"[SERVICE] Loaded model, scaler, and {len(self.feature_cols)} features.")

        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {e}")
            raise RuntimeError("Model service initialization failed.")

    def create_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculates rolling mean features, similar to data_preprocessing.py."""
        
        # Identify original sensor columns (those without '_roll_mean_')
        # We assume original sensor names contain 'Sensor_'
        original_sensor_cols = [
            col for col in df.columns 
            if 'Sensor' in col and '_roll_mean_' not in col
        ]
        
        # Create rolling mean features
        for col in original_sensor_cols:
            new_col_name = f'{col}_roll_mean_{window}'
            # Since this is a single engine's data, we just apply rolling mean directly
            df[new_col_name] = df[col].rolling(window=window, min_periods=1).mean()
        
        return df

    def prepare_data_and_predict(self, raw_df: pd.DataFrame) -> float:
        """
        Prepares raw sensor data for prediction:
        1. Calculates Rolling Means.
        2. Scales data using the loaded scaler.
        3. Generates the final sequence (last 50 cycles).
        4. Predicts RUL.
        """
        
        # 1. Feature Engineering (Rolling Means)
        df_featured = self.create_rolling_features(raw_df.copy())
        
        # 2. Scaling
        # We only need the columns that were used for training (self.feature_cols)
        features_to_scale_and_use = [col for col in self.feature_cols if col in df_featured.columns]
        
        if len(features_to_scale_and_use) != len(self.feature_cols):
             raise ValueError("Input data is missing expected features after rolling mean calculation.")
        
        df_featured[features_to_scale_and_use] = self.scaler.transform(df_featured[features_to_scale_and_use])

        # 3. Sequence Generation
        # Get the last 'window_length' cycles
        sequence = df_featured[features_to_scale_and_use].values[-self.window_length:]

        # Reshape for LSTM: (1, window_length, n_features)
        X = sequence.reshape(1, self.window_length, len(features_to_scale_and_use))

        # 4. Prediction
        prediction_scaled = self.model.predict(X, verbose=0)[0][0]

        # The model predicts the *scaled* RUL, but since the RUL target was clipped 
        # (MinMax scaling from 0 to 1), we return the scaled value here. 
        # For a standard RUL system, you might want to inverse-scale it, but 
        # since the target RUL was also scaled 0-1, the prediction is also 0-1.
        
        # However, it's better to return the RUL in its original cycle space (up to 125).
        # To inverse-scale the RUL, we need the RUL max value (125) and min value (0).
        # The true inverse transform is complex, but since RUL was a simple clip:
        # RUL = Scaled_RUL * (RUL_Limit - 0) + 0 
        
        # Note: A proper inverse transform requires the scaler object fitted only on RUL, 
        # but for this specific target, we approximate the inverse of the RUL target scaling.
        RUL_LIMIT = 125 
        predicted_rul = prediction_scaled * RUL_LIMIT 

        # Clip final prediction at the RUL limit
        predicted_rul = max(0, predicted_rul) 

        return float(predicted_rul)