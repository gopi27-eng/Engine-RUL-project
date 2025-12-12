import os
import sqlite3
import io
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template

# --- Configuration ---
# CORRECTED PATHS based on user's confirmed file structure
MODEL_PATH = 'artifacts/model/best_rul_model.keras' 
SCALER_PATH = 'artifacts/scaler/minmax_scaler.pkl' 

# CRITICAL FIX: Use /tmp/ for database to ensure write permissions on Render/Cloud
DB_PATH = '/tmp/prediction_history.db' 
WINDOW_LENGTH = 50 # Sequence window length for the LSTM model
RMSE_CONFIDENCE = 14.95 # Placeholder for Model Confidence metric

class RULPredictionService:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, db_path=DB_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.db_path = db_path
        self.window_length = WINDOW_LENGTH
        
        # Artifacts will be loaded here
        self.model = None
        self.scaler = None
        
        self._load_artifacts()
        self._setup_database()

    def _load_artifacts(self):
        """Loads the trained Keras model and the MinMaxScaler."""
        try:
            # Load Keras Model
            self.model = load_model(self.model_path)
            
            # Load Scaler
            self.scaler = joblib.load(self.scaler_path)
            print("INFO: Model and Scaler artifacts loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load artifacts. Check paths and file existence: {e}")
            raise RuntimeError(f"Model service failed to start: {e}")

    def _setup_database(self):
        """Creates the SQLite table for logging predictions if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Prediction_History (
                    Prediction_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Timestamp TEXT NOT NULL,
                    Engine_ID INTEGER NOT NULL,
                    Cycle_Max INTEGER NOT NULL,
                    Predicted_RUL REAL NOT NULL,
                    Model_Confidence REAL
                );
            """)
            conn.commit()
            conn.close()
            print(f"[DB] Prediction_History database initialized at {self.db_path}.")
        except Exception as e:
            print(f"ERROR: Failed to set up database at {self.db_path}: {e}")
            raise RuntimeError(f"Database setup failed: {e}")

    def log_prediction_to_db(self, engine_id: int, cycle_max: int, rul: float, confidence: float = RMSE_CONFIDENCE):
        """Logs a prediction result to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute("""
                INSERT INTO Prediction_History 
                (Timestamp, Engine_ID, Cycle_Max, Predicted_RUL, Model_Confidence) 
                VALUES (?, ?, ?, ?, ?);
            """, (timestamp_str, engine_id, cycle_max, rul, confidence))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"WARNING: Failed to log prediction: {e}")

    def prepare_data_and_predict(self, input_df: pd.DataFrame) -> float:
        """Scales, sequences data, and makes the RUL prediction."""
        
        # 1. Scaling the data
        # Assuming sensor columns are indices 5 to 25 (21 sensors)
        cols_to_scale = input_df.columns[5:26]
        
        if len(cols_to_scale) != 21: # Double check the feature count
             raise ValueError(f"Scaling feature count mismatch: Expected 21 sensors, found {len(cols_to_scale)}.")

        scaled_data = self.scaler.transform(input_df[cols_to_scale])
        
        # 2. Sequence creation (using only the last 'window_length' cycles)
        if len(scaled_data) < self.window_length:
            raise ValueError(f"Insufficient data ({len(scaled_data)} cycles). Need exactly {self.window_length} cycles for prediction.")
            
        # Get the last sequence of the scaled data
        sequence = scaled_data[-self.window_length:]
        
        # Reshape for LSTM: (1, window_length, n_features)
        X_test_sequence = sequence.reshape(1, self.window_length, sequence.shape[1])
        
        # 3. Prediction
        predicted_rul = self.model.predict(X_test_sequence)[0][0]
        
        # RUL must be non-negative
        return max(0, float(predicted_rul))

# --- Flask Application Initialization ---
app = Flask(__name__, template_folder='templates')

# Initialize the service instance (This runs once when the Flask app starts)
try:
    service = RULPredictionService()
except RuntimeError:
    # Service failed to initialize (e.g., artifacts missing, DB error)
    service = None

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def predictor_ui():
    result = None
    if service is None:
        return render_template('index.html', error="Model service is currently offline. Check artifact paths and server logs."), 503
        
    if request.method == 'POST':
        try:
            # 1. Get uploaded file
            uploaded_file = request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '':
                raise ValueError("No file selected.")
                
            # Read the file content into a stream
            file_stream = io.StringIO(uploaded_file.stream.read().decode('latin1'))

            # 2. Read data (FORCED ENCODING FIX: reading as latin1)
            # This is the most reliable way to handle CSVs saved on Windows/Excel.
            file_stream.seek(0)
            input_df = pd.read_csv(file_stream)
            
            # Basic data validation
            if input_df.shape[1] != 26:
                raise ValueError(f"Data format invalid: Expected 26 columns, received {input_df.shape[1]}.")
            if 'Engine_No' not in input_df.columns or 'Cycle' not in input_df.columns:
                raise ValueError("Data format invalid: Missing 'Engine_No' or 'Cycle' columns.")

            # 3. Ensure data is sorted by cycle for correct sequencing
            input_df = input_df.sort_values(by=['Engine_No', 'Cycle']).reset_index(drop=True)

            # 4. Call the prediction service
            predicted_rul = service.prepare_data_and_predict(input_df)
            
            # --- Logging the Result ---
            engine_id = int(input_df['Engine_No'].iloc[0])
            cycle_max = int(input_df['Cycle'].max())
            service.log_prediction_to_db(engine_id, cycle_max, predicted_rul, confidence=RMSE_CONFIDENCE)
            # ---------------------------

            # 5. Prepare the result for the template
            result = {
                'Engine ID': engine_id,
                'Max Cycle': cycle_max,
                'Predicted RUL': f"{predicted_rul:.2f} Cycles",
                'Confidence Metric (RMSE)': f"{RMSE_CONFIDENCE:.2f} (Model RMSE)"
            }
            
        except Exception as e:
            print(f"Runtime Prediction Error: {e}") 
            return render_template('index.html', error=f"Prediction Error: {e}")

    return render_template('index.html', result=result)

# --- Entry Point for Gunicorn/Web Server ---
if __name__ == '__main__':
    # This block is typically for local testing only
    app.run(debug=True, host='0.0.0.0', port=5000)