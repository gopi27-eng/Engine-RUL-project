# src/data_preprocessing.py

import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
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

# --- Main Data Preprocessor Class ---
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.column_names = config['column_names']
        self.dropped_columns = config['dropped_columns']
        self.model_constants = config['model_constants']
        self.local_data_paths = config['local_data_paths']
        self.artifact_paths = config['artifact_paths']
        
        # Will be defined after feature selection
        self.feature_cols = []
        
        # Ensure output directories exist
        os.makedirs(self.artifact_paths['processed_data_dir'], exist_ok=True)
        os.makedirs(self.artifact_paths['scaler_dir'], exist_ok=True)

    def calculate_RUL(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Remaining Useful Life (RUL) for each cycle."""
        
        rul_limit = self.model_constants['rul_limit']
        
        max_cycle_df = df.groupby('Engine_No')['Cycle'].max().reset_index()
        max_cycle_df.columns = ['Engine_No', 'Max_Cycle']
        
        df = df.merge(max_cycle_df, on='Engine_No', how='left')
        df['RUL'] = df['Max_Cycle'] - df['Cycle']
        df.drop('Max_Cycle', axis=1, inplace=True)
        
        # Apply RUL limit (clipping)
        df['RUL'] = df['RUL'].apply(lambda x: min(x, rul_limit))
        
        logger.info(f"RUL calculated and clipped at {rul_limit} cycles.")
        return df

    def feature_selection_and_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary and constant features but retains Engine_No and Cycle."""
        
        # 1. Find all sensor/op columns NOT in dropped_columns
        sensor_op_cols = [
            col for col in self.column_names
            if col not in ['Engine_No', 'Cycle'] and col not in self.dropped_columns
        ]

        # 2. Identify all columns that MUST be kept in the DataFrame
        cols_to_keep = ['Engine_No', 'Cycle'] + sensor_op_cols
        if 'RUL' in df.columns:
            cols_to_keep.append('RUL')
        
        # 3. Select the data, dropping only the unnecessary columns
        df = df[cols_to_keep].copy()
        
        # 4. Define the list of FEATURES (only the columns that get scaled)
        self.feature_cols = sensor_op_cols 

        logger.info(f"Features selected. Initial feature count: {len(self.feature_cols)}")
        return df
    
    # --- FIXED METHOD: FEATURE ENGINEERING (ROLLING MEAN) ---
    def create_rolling_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Calculates rolling mean features for sensor data."""
        
        # Safely retrieve rolling window size from config
        window = self.model_constants.get('rolling_window', 5) 
        
        # FIX 1: Only select ORIGINAL sensor columns that do not already contain '_roll_mean_'
        sensor_cols_for_roll = [
            col for col in self.feature_cols 
            if 'Sensor' in col and '_roll_mean_' not in col
        ]
        
        logger.info(f"Creating rolling mean features for {len(sensor_cols_for_roll)} sensors using window={window}.")

        # Create new rolling mean columns for all relevant sensors
        for col in sensor_cols_for_roll:
            new_col_name = f'{col}_roll_mean_{window}'
            
            df[new_col_name] = df.groupby('Engine_No')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # FIX 2: Only update the self.feature_cols list during the TRAIN step (is_train=True).
            # This prevents the test data step from adding the new features to the global list 
            # and causing nested rolling in the subsequent call.
            if is_train:
                 self.feature_cols.append(new_col_name)
            
        logger.info(f"Feature count (currently being tracked): {len(self.feature_cols)}")
        return df

    def scale_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Fits MinMaxScaler on training data and transforms both train and test sets."""
        
        # We must only scale the feature columns, not Engine_No, Cycle, or RUL
        features_to_scale = self.feature_cols

        scaler = MinMaxScaler()
        
        # Fit on TRAIN data and transform both sets
        train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
        test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
        
        # Save the fitted scaler
        scaler_path = os.path.join(self.artifact_paths['scaler_dir'], 'minmax_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info(f"MinMaxScaler fitted and saved to {scaler_path}")
        return train_df, test_df, scaler

    def initiate_data_preprocessing(self):
        """The main execution method for the preprocessing pipeline."""
        
        # 1. Data Ingestion (Load raw data)
        train_path = os.path.join(self.local_data_paths['root_dir'], self.local_data_paths['train_file'])
        test_path = os.path.join(self.local_data_paths['root_dir'], self.local_data_paths['test_file'])
        
        train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=self.column_names)
        test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=self.column_names)
        
        # Load the true RUL values for the test set (needed for final analysis only)
        true_rul_path = os.path.join(self.local_data_paths['root_dir'], self.local_data_paths['true_rul_file'])
        true_rul_df = pd.read_csv(true_rul_path, sep=r"\s+", header=None, names=['RUL']).dropna(axis=1)

        # 2. RUL Calculation (for Training Data)
        train_df = self.calculate_RUL(train_df)
        
        # 3. Feature Selection & Cleanup 
        train_df = self.feature_selection_and_cleanup(train_df)
        test_df = self.feature_selection_and_cleanup(test_df)
        
        # 4. Feature Engineering (Rolling Means) - PASSING THE is_train FLAG
        train_df = self.create_rolling_features(train_df, is_train=True)
        test_df = self.create_rolling_features(test_df, is_train=False)

        # 5. Scaling
        train_df_scaled, test_df_scaled, _ = self.scale_data(train_df, test_df)

        # 6. Saving Processed Data
        output_train_path = os.path.join(self.artifact_paths['processed_data_dir'], 'processed_train.csv')
        output_test_path = os.path.join(self.artifact_paths['processed_data_dir'], 'processed_test.csv')
        output_true_rul_path = os.path.join(self.artifact_paths['processed_data_dir'], 'true_rul.csv')

        train_df_scaled.to_csv(output_train_path, index=False)
        test_df_scaled.to_csv(output_test_path, index=False)
        true_rul_df.to_csv(output_true_rul_path, index=False)
        
        logger.info(f"Processed data saved to {self.artifact_paths['processed_data_dir']}.")


# --- Main Execution ---
if __name__ == "__main__":
    
    logger.info("--- Starting Data Preprocessing Pipeline ---")
    config = load_config()
    
    # Add optional keys if not in config (for robustness)
    if 'rolling_window' not in config['model_constants']:
        config['model_constants']['rolling_window'] = 5
        
    preprocessor = DataPreprocessor(config)
    preprocessor.initiate_data_preprocessing()
    logger.info("Data Preprocessing pipeline step finished successfully.")