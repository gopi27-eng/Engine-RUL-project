# src/analysis_and_reporting.py

import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Assuming utility files exist
from utils.logger import logger

# --- Utility Functions (Duplicated for modularity) ---
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

# --- Core Analysis Class ---
class PerformanceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.artifact_paths = config['artifact_paths']
        self.model_constants = config['model_constants']
        self.window_length = self.model_constants['window_length']
        self.rul_limit = self.model_constants['rul_limit']
        
        # Load artifacts
        try:
            self.model = load_model(os.path.join(self.artifact_paths['model_dir'], 'best_rul_model.keras'))
            logger.info("Trained LSTM Model loaded for analysis.")
            
            self.processed_test_df = pd.read_csv(os.path.join(self.artifact_paths['processed_data_dir'], 'processed_test.csv'))
            self.true_rul_df = pd.read_csv(os.path.join(self.artifact_paths['processed_data_dir'], 'true_rul.csv'))
            logger.info("Processed test data and True RULs loaded.")
            
        except FileNotFoundError as e:
            logger.error(f"FATAL: Required artifact not found. Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            sys.exit(1)

    def generate_sequences(self) -> tuple[np.ndarray, list]:
        """Creates the 3D sequence data for the test set predictions."""
        
        all_test_sequences = []
        num_test_windows_list = []
        
        # Determine the feature columns (Engine_No and Cycle are excluded)
        feature_cols = [col for col in self.processed_test_df.columns if col not in ['Engine_No', 'Cycle']]

        # Group data by Engine ID
        grouped = self.processed_test_df.groupby('Engine_No')
        
        for engine_id, df_engine in grouped:
            # We only need the last 'window_length' cycles for prediction
            data_values = df_engine[feature_cols].values
            
            if data_values.shape[0] >= self.window_length:
                # Use the last sequence
                sequence = data_values[-self.window_length:]
            else:
                # Pad if engine life is shorter than the window (unlikely for FD001 test set)
                padding_needed = self.window_length - data_values.shape[0]
                padding = np.zeros((padding_needed, data_values.shape[1]))
                sequence = np.vstack([padding, data_values])
                
            all_test_sequences.append(sequence)
            num_test_windows_list.append(engine_id) # Store the engine ID instead of window count

        # Convert to 3D array (N_engines, WINDOW_LENGTH, N_FEATURES)
        processed_test_data = np.array(all_test_sequences)
        logger.info(f"Test data transformed into sequences. Shape: {processed_test_data.shape}")
        return processed_test_data, num_test_windows_list

    def predict_and_analyze(self) -> pd.DataFrame:
        """Runs predictions, compares to true RUL, and creates the final analysis DataFrame."""
        
        # 1. Generate sequences for prediction
        test_sequences, engine_ids = self.generate_sequences()
        
        # 2. Make predictions
        logger.info("Generating predictions for all test engines...")
        # Predictions are generated for the last sequence of each engine
        predictions = self.model.predict(test_sequences)
        
        # Extract RUL (which is the first element of the output vector)
        predicted_ruls = predictions[:, 0]
        
        # Apply the RUL cap
        predicted_ruls = np.minimum(predicted_ruls, self.rul_limit)
        
        # 3. Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Engine_No': engine_ids,
            'True_RUL': self.true_rul_df['RUL'].values, # True RULs correspond directly to the engine_ids list
            'Predicted_RUL': predicted_ruls
        })
        
        # 4. Calculate error metrics
        comparison_df['Error'] = comparison_df['Predicted_RUL'] - comparison_df['True_RUL']
        
        # 5. Save the analysis data
        output_path = os.path.join(self.artifact_paths['analysis_dir'], 'final_predictions.csv')
        os.makedirs(self.artifact_paths['analysis_dir'], exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        
        logger.info(f"Final prediction analysis data saved to {output_path}")
        return comparison_df

    def visualize_performance(self, df: pd.DataFrame):
        """Generates a scatter plot of True vs. Predicted RUL."""
        
        plt.figure(figsize=(10, 8))
        
        # Scatter Plot of Predicted vs. True RUL
        plt.scatter(df['True_RUL'], df['Predicted_RUL'], alpha=0.6, edgecolors='w', linewidth=0.5, color='#1f77b4')
        
        # Line of perfect prediction (y=x)
        min_rul = min(df['True_RUL'].min(), df['Predicted_RUL'].min())
        max_rul = max(df['True_RUL'].max(), df['Predicted_RUL'].max())
        plt.plot([min_rul, max_rul], [min_rul, max_rul], 'r--', label='Perfect Prediction (y=x)')
        
        plt.xlabel('True Remaining Useful Life (RUL) [cycles]', fontsize=12)
        plt.ylabel('Predicted Remaining Useful Life (RUL) [cycles]', fontsize=12)
        plt.title('True vs. Predicted RUL on Test Set (FD001)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(self.artifact_paths['analysis_dir'], 'rul_prediction_scatter.png')
        plt.savefig(plot_path)
        logger.info(f"Performance scatter plot saved to {plot_path}")
        
        # Show a detail plot (True RUL vs. Prediction for first 10 engines)
        plt.figure(figsize=(12, 6))
        
        sample_df = df.head(10)
        
        plt.plot(sample_df['Engine_No'], sample_df['True_RUL'], marker='o', linestyle='-', color='green', label='True RUL')
        plt.plot(sample_df['Engine_No'], sample_df['Predicted_RUL'], marker='x', linestyle='--', color='red', label='Predicted RUL')
        
        plt.xlabel('Engine ID', fontsize=12)
        plt.ylabel('RUL (cycles)', fontsize=12)
        plt.title('True vs. Predicted RUL for 10 Sample Engines', fontsize=14)
        plt.xticks(sample_df['Engine_No'])
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        
        # Save the plot
        sample_plot_path = os.path.join(self.artifact_paths['analysis_dir'], 'rul_prediction_sample_line.png')
        plt.savefig(sample_plot_path)
        logger.info(f"Sample engine line plot saved to {sample_plot_path}")


# --- Example Execution ---
if __name__ == "__main__":
    
    logger.info("--- Starting Analysis and Reporting Component ---")
    
    # 1. Load config and initialize
    config = load_config()
    # Ensure analysis directory exists and is in config (assuming it's artifacts/analysis)
    if 'analysis_dir' not in config['artifact_paths']:
        config['artifact_paths']['analysis_dir'] = 'artifacts/analysis'
        
    analyzer = PerformanceAnalyzer(config)
    
    # 2. Run prediction and comparison
    final_df = analyzer.predict_and_analyze()
    
    # 3. Generate visualizations
    analyzer.visualize_performance(final_df)
    
    logger.info("Analysis and Reporting Component finished successfully.")