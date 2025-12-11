import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle  # <-- NEW: Import pickle for saving the scaler
from sklearn.preprocessing import MinMaxScaler

# Assuming utility files exist
from utils.logger import logger

# --- Configuration Loader (Utility function - duplicated for modularity) ---
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


# --- Core Preprocessing Class ---
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.artifact_paths = config['artifact_paths']
        self.model_constants = config['model_constants']
        self.dropped_cols = config['dropped_columns']
        self.feature_cols = []
        
        # Ensure all necessary artifact directories exist
        os.makedirs(self.artifact_paths['processed_data_dir'], exist_ok=True)
        os.makedirs(self.artifact_paths['scaler_dir'], exist_ok=True) # <-- NEW: Ensure scaler directory exists
        logger.info(f"Created processed artifact directory: {self.artifact_paths['processed_data_dir']}")

    # ... (calculate_rul and feature_selection_and_cleanup methods remain the same) ...

    def calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the RUL and applies the piecewise linear cap (RUL_LIMIT)."""
        logger.info("Calculating RUL and applying cap.")
        
        # 1. Find max cycle for each engine
        max_cycle_df = df.groupby('Engine_No')['Cycle'].max().reset_index()
        max_cycle_df.columns = ['Engine_No', 'Max_Cycle']
        df = df.merge(max_cycle_df, on='Engine_No', how='left')
        
        # 2. Calculate RUL and apply cap
        df['RUL'] = df['Max_Cycle'] - df['Cycle']
        df.drop('Max_Cycle', axis=1, inplace=True)
        
        rul_limit = self.model_constants['rul_limit']
        df['RUL'] = df['RUL'].apply(lambda x: min(x, rul_limit))
        
        logger.info(f"RUL calculated. Cap applied at {rul_limit} cycles.")
        return df

    def feature_selection_and_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops identified constant columns and defines the final feature set."""
        
        cols_to_drop = self.dropped_cols
        
        logger.info(f"Dropping {len(cols_to_drop)} constant/redundant columns.")
        
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        
        # Define the final feature set for scaling (all non-Engine_No, Cycle, RUL columns)
        self.feature_cols = [
            col for col in df.columns 
            if col not in ['Engine_No', 'Cycle', 'RUL']
        ]
        
        logger.info(f"Final feature count for scaling: {len(self.feature_cols)}")
        return df


    def scale_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        """Applies Min-Max Scaling based ONLY on the training data."""
        
        scaler = MinMaxScaler()
        feature_cols = self.feature_cols
        
        logger.info("Initializing and fitting MinMaxScaler on training data.")
        
        # Fit on Training Data
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        
        # Transform Test Data
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        logger.info("Data scaling complete for both train and test sets.")
        return train_df, test_df, scaler

    def initiate_data_preprocessing(self, raw_train_path: str, raw_test_path: str, raw_rul_path: str) -> tuple[str, str]:
        """Orchestrates the entire preprocessing pipeline."""
        
        logger.info("Starting Data Preprocessing component.")
        
        # 1. Load Raw Data
        column_names = self.config['column_names']
        train_df = pd.read_csv(raw_train_path, sep=r"\s+", header=None, names=column_names)
        test_df = pd.read_csv(raw_test_path, sep=r"\s+", header=None, names=column_names)
        rul_df = pd.read_csv(raw_rul_path, sep=r"\s+", header=None, names=['RUL']).dropna(axis=1)
        
        logger.info(f"Raw data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # 2. RUL Calculation 
        train_df = self.calculate_rul(train_df)

        # 3. Feature Selection & Cleanup
        train_df = self.feature_selection_and_cleanup(train_df)
        test_df = self.feature_selection_and_cleanup(test_df)
        
        # 4. Scaling
        train_df_scaled, test_df_scaled, scaler = self.scale_data(train_df, test_df)
        
        # 5. Save Processed Artifacts AND SCALER
        
        # --- NEW: Save the fitted MinMaxScaler object ---
        scaler_path = os.path.join(self.artifact_paths['scaler_dir'], 'minmax_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"MinMaxScaler saved to {scaler_path}")
        # -----------------------------------------------

        processed_train_path = os.path.join(self.artifact_paths['processed_data_dir'], 'processed_train.csv')
        processed_test_path = os.path.join(self.artifact_paths['processed_data_dir'], 'processed_test.csv')
        
        final_cols = ['Engine_No', 'Cycle', 'RUL'] + self.feature_cols
        
        train_df_scaled[final_cols].to_csv(processed_train_path, index=False)
        test_df_scaled[['Engine_No', 'Cycle'] + self.feature_cols].to_csv(processed_test_path, index=False)
        rul_df.to_csv(os.path.join(self.artifact_paths['processed_data_dir'], 'true_rul.csv'), index=False)

        logger.info(f"Processed training data saved to {processed_train_path}")
        logger.info(f"Processed test data saved to {processed_test_path}")

        return processed_train_path, processed_test_path

# --- Example Execution ---
if __name__ == "__main__":
    
    config = load_config()
    
    # Assume files are in the local raw data artifact folder
    raw_dir = config['artifact_paths']['raw_data_dir']
    
    raw_train = os.path.join(raw_dir, 'train_FD001.txt')
    raw_test = os.path.join(raw_dir, 'test_FD001.txt')
    raw_rul = os.path.join(raw_dir, 'RUL_FD001.txt')
    
    if not all(os.path.exists(f) for f in [raw_train, raw_test, raw_rul]):
        logger.warning("RAW data files not found in 'artifacts/raw_data'. Please run data_ingestion.py locally first.")
        sys.exit(0)

    preprocessor = DataPreprocessor(config)
    processed_train_path, processed_test_path = preprocessor.initiate_data_preprocessing(
        raw_train, raw_test, raw_rul
    )
    
    logger.info("Data Preprocessing pipeline step finished successfully.")