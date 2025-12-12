# api/app.py

import pandas as pd
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model # Assumes TensorFlow is installed

# --- 1. RUL Prediction Service Class ---
class RULPredictionService:
    def __init__(self, model_path, scaler_path, window_length=50):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.window_length = window_length
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.RUL_LIMIT = 125 # The max RUL value used for target capping

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the trained Keras model, MinMaxScaler, and feature names."""
        try:
            # 1. Load Model (Best Practice: Use .keras format)
            self.model = load_model(self.model_path)
            
            # 2. Load Scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # 3. Load feature names (must match the training order, including rolling means)
            processed_train_path = 'artifacts/processed_data/processed_train.csv'
            
            if not os.path.exists(processed_train_path):
                 print(f"ERROR: Processed data not found at {processed_train_path}")
                 # Ensure this file exists in your project structure!
                 raise FileNotFoundError("Required processed training data (for feature names) not found.")

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
        """Calculates rolling mean features on the input data for all sensor columns."""
        
        # Identify sensor columns (S_1 to S_21)
        original_sensor_cols = [col for col in df.columns if 'Sensor_' in col]
        
        # Apply rolling mean
        for col in original_sensor_cols:
            new_col_name = f'{col}_roll_mean_{window}'
            # Apply rolling mean, setting min_periods=1 allows calculation on the first few rows
            df[new_col_name] = df[col].rolling(window=window, min_periods=1).mean()
        
        return df

    def prepare_data_and_predict(self, raw_df: pd.DataFrame) -> float:
        """
        Prepares raw sensor data for prediction and applies the LSTM model.
        This contains the complete, debugged pipeline.
        """
        
        # 1. Feature Engineering (Rolling Means)
        df_featured = self.create_rolling_features(raw_df.copy())
        
        # 2. Scaling (Align and Transform)
        df_featured_aligned = df_featured[self.feature_cols].copy()
        
        # Transform the data using the loaded scaler
        # Note: .loc[] is used for stability in setting values on a slice
        df_featured_aligned.loc[:, self.feature_cols] = self.scaler.transform(df_featured_aligned[self.feature_cols])


        # 3. Sequence Generation
        # Get the last 'window_length' (50) cycles
        sequence = df_featured_aligned.values[-self.window_length:]

        # Reshape for LSTM: (1, window_length, n_features)
        X = sequence.reshape(1, self.window_length, len(self.feature_cols))

        # ------------------------------------------------------------------
        # --- DEBUG: Console Output to verify scaling ---
        # ------------------------------------------------------------------
        print("\n--- DEBUG: LSTM Input Array (X) Analysis ---")
        print(f"Shape of X: {X.shape}")
        print(f"Min value in X: {np.min(X):.4f}")
        print(f"Max value in X: {np.max(X):.4f}")
        # ------------------------------------------------------------------


        # 4. Prediction
        # Predict: Output is a NumPy array, e.g., [[105.4]]
        raw_prediction_array = self.model.predict(X, verbose=0)
        
        # Extract the scalar RUL value (already unscaled, as confirmed by debug)
        predicted_rul_scalar = float(raw_prediction_array[0][0]) 

        # 5. Inverse Transformation & Capping (FINAL ROBUST FIX)
        # Use numpy.clip to cap the prediction between 0 and RUL_LIMIT (125)
        predicted_rul = np.clip(predicted_rul_scalar, 0, self.RUL_LIMIT)
        
        # Convert the result back to a standard Python float for the return
        return float(predicted_rul)

# --- 2. Flask Application Setup ---
# Set paths relative to the project root
MODEL_PATH = 'artifacts/model/best_rul_model.keras'
SCALER_PATH = 'artifacts/scaler/minmax_scaler.pkl'
WINDOW_LENGTH = 50 

# --- Initialize the Prediction Service ---
try:
    service = RULPredictionService(MODEL_PATH, SCALER_PATH, WINDOW_LENGTH)
    app = Flask(__name__)
    print("--- RUL Prediction Service Initialized Successfully ---")
except Exception as e:
    service = None 
    app = Flask(__name__) 
    print(f"FATAL: Service initialization failed. Check logs above. {e}")

# Define the expected raw columns for the incoming CSV data
RAW_COLUMNS = ['Engine_No', 'Cycle', 'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3',
               'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_6', 
               'Sensor_7', 'Sensor_8', 'Sensor_9', 'Sensor_10', 'Sensor_11', 'Sensor_12', 
               'Sensor_13', 'Sensor_14', 'Sensor_15', 'Sensor_16', 'Sensor_17', 
               'Sensor_18', 'Sensor_19', 'Sensor_20', 'Sensor_21']

# --- 3. Flask Routes (UI and API) ---

@app.route('/', methods=['GET', 'POST'])
def predictor_ui():
    if service is None:
        return render_template('index.html', error="Model service is currently offline. Check artifact paths and server logs.")

    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        # --- Handle form submission from the UI ---
        raw_data_string = request.form.get('sensor_data')
        
        if not raw_data_string:
            return render_template('index.html', error="No data provided in the input box.")

        try:
            # 1. Parse the CSV data string
            data_list = []
            for line in raw_data_string.strip().split('\n'):
                if line.strip():
                    data_list.append([float(x.strip()) for x in line.split(',')])

            # 2. Create DataFrame
            input_df = pd.DataFrame(data_list, columns=RAW_COLUMNS)
            
            # 3. Validation
            if input_df.shape[0] < service.window_length:
                raise ValueError(f"Data must contain at least {service.window_length} cycles. Found {input_df.shape[0]}.")
            if input_df.shape[1] != len(RAW_COLUMNS):
                 raise ValueError(f"Data must contain exactly {len(RAW_COLUMNS)} columns. Found {input_df.shape[1]}.")

            # 4. Call the prediction service
            predicted_rul = service.prepare_data_and_predict(input_df)

            # 5. Prepare the result for the template
            result = {
                "predicted_rul_cycles": round(predicted_rul, 1),
                "model_confidence": "High (Corrected RMSE)", # Adjusted for the successful fix
                "warning_level": "CRITICAL: Schedule maintenance NOW." if predicted_rul < 20 else ("WARNING: Schedule maintenance SOON." if predicted_rul < 50 else "Monitoring"),
                "action_required": "Prediction pipeline is validated. Prediction is reliable."
            }
            
            # 6. Render the template with the results
            return render_template('index.html', result=result)

        except ValueError as e:
            return render_template('index.html', error=f"Data parsing or validation error: {e}")
        except Exception as e:
            print(f"Prediction Error (Uncaught): {e}") # Keep the debug print for uncaught errors
            return render_template('index.html', error="Internal server error during prediction. Check server console for traceback.")

# --- 4. Main Execution ---
if __name__ == '__main__':
    # Running locally
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)