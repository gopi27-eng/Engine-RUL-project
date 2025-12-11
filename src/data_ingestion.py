# src/data_ingestion.py

import os
import sys
import yaml
import pandas as pd
from datetime import datetime
import boto3 # Keep for future S3 implementation
from botocore.exceptions import NoCredentialsError, ClientError

# Assuming utils/logger.py exists
from utils.logger import logger 

# --- Configuration Loader (Utility function - for modularity) ---
def load_config(config_path='config/config.yaml'):
    """Loads configuration from the specified YAML file."""
    try:
        # Ensure path is relative to the current working directory
        config_path = os.path.join(os.getcwd(), config_path)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration file loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

# --- Data Loading Utility (Local Disk) ---
def read_data_from_local_disk(local_path: str, col_names: list, is_rul_file: bool = False) -> pd.DataFrame:
    """Reads data from the local filesystem using the specified column names."""
    try:
        logger.info(f"Reading local data from: {local_path}")
        
        # Read the space-separated file
        df = pd.read_csv(local_path, sep=r"\s+", header=None, names=col_names)
        
        # The FD001 files often have NaN columns at the end due to trailing spaces, which must be dropped
        if not is_rul_file:
            # For train/test files, drop the two final NaN columns
            df = df.iloc[:, :-2]
        
        logger.info(f"Successfully read local data. DataFrame shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"FATAL ERROR: Local file not found at {local_path}. Check config/config.yaml or the path.")
        raise
    except Exception as e:
        logger.error(f"Error reading local data from {local_path}: {e}")
        raise

# --- Main Data Ingestion Class ---
class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.artifact_paths = config['artifact_paths']
        # Dynamically fetch the root path from config
        self.local_data_root = config['local_data_paths']['root_dir'] 
        self.s3_config = config['s3_data_paths']
        
        # Create necessary artifact directories
        os.makedirs(self.artifact_paths['raw_data_dir'], exist_ok=True)
        logger.info(f"Created artifact directory: {self.artifact_paths['raw_data_dir']}")

    def _get_local_file_paths(self) -> dict:
        """Constructs the full local paths to the raw data files using the config root."""
        root = self.local_data_root
        return {
            'train': os.path.join(root, 'train_FD001.txt'),
            'test': os.path.join(root, 'test_FD001.txt'),
            'rul': os.path.join(root, 'RUL_FD001.txt')
        }
    
    def _save_raw_data(self, df: pd.DataFrame, artifact_name: str, is_rul_file: bool = False) -> str:
        """Saves a DataFrame to the raw artifacts directory."""
        output_path = os.path.join(self.artifact_paths['raw_data_dir'], artifact_name)
        
        # Use space separator to maintain the original file format structure
        df.to_csv(output_path, index=False, header=False, sep=' ')
             
        logger.info(f"Raw data artifact saved to {output_path}.")
        return output_path

    def initiate_data_ingestion(self, mode='local'):
        """
        Orchestrates the data ingestion process.
        
        Args:
            mode (str): 'local' to read from local drive (DEV/TEST) or 's3' (PROD).
        """
        logger.info(f"Starting Data Ingestion component in '{mode}' mode.")
        
        try:
            col_names = self.config['column_names']
            
            if mode == 'local':
                local_paths = self._get_local_file_paths()
                
                # Load data from local files
                train_df = read_data_from_local_disk(local_paths['train'], col_names)
                test_df = read_data_from_local_disk(local_paths['test'], col_names)
                # RUL file has only one column
                rul_df = read_data_from_local_disk(local_paths['rul'], ['RUL'], is_rul_file=True)
                
            elif mode == 's3':
                logger.warning("S3 mode is selected. You must implement Boto3 download logic here.")
                # In a real environment, you would use Boto3 here:
                # s3 = boto3.client('s3')
                # s3.download_file(self.s3_config['bucket_name'], self.s3_config['train_data'], TRAIN_TEMP_PATH)
                raise NotImplementedError("S3 data loading not implemented in this version. Use 'local' mode.")
            
            else:
                raise ValueError(f"Invalid ingestion mode: {mode}. Must be 'local' or 's3'.")

            # --- 2. Save Artifacts ---
            train_file = self._save_raw_data(train_df, 'train_FD001.txt')
            test_file = self._save_raw_data(test_df, 'test_FD001.txt')
            rul_file = self._save_raw_data(rul_df, 'RUL_FD001.txt', is_rul_file=True)
            
            logger.info("Data Ingestion complete.")
            
            return train_file, test_file, rul_file

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e

# --- Example Execution (for your local testing) ---
if __name__ == "__main__":
    # Ensure __init__.py exists in src/ to handle imports
    open("src/__init__.py", "a").close() 
    
    # 1. Load config
    config = load_config()
    
    # 2. Run ingestion
    ingestion_obj = DataIngestion(config)
    
    try:
        # Running in 'local' mode to read from C:\ sync data path
        train_path, test_path, rul_path = ingestion_obj.initiate_data_ingestion(mode='local')
        logger.info(f"Ready for Data Preprocessing with files: {train_path}, {test_path}, {rul_path}")
    except Exception as e:
        # Error handling will log the file not found error if the path is wrong
        sys.exit(1)