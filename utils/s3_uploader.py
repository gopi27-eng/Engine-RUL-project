# utils/s3_uploader.py

import boto3
import os
import sys
import yaml
from datetime import datetime
from botocore.exceptions import NoCredentialsError, ClientError

# --- IMPORTANT: Ensure 'utils/logger.py' exists in your project ---
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
        return None

# --- S3 Uploader Class ---
class S3Uploader:
    """
    A utility class for managing file uploads to an AWS S3 bucket.
    It relies on a boto3 Session to securely handle credentials 
    (from environment variables, AWS CLI config, or IAM role).
    """
    def __init__(self, bucket_name: str, region_name: str = 'us-east-1'):
        self.bucket_name = bucket_name
        
        # boto3.Session automatically looks for credentials in the environment 
        # (which is where a loaded .env file puts them) or in AWS config files.
        self.session = boto3.Session(region_name=region_name)
        self.s3 = self.session.client('s3')
        
        logger.info(f"S3 Uploader initialized for bucket: {self.bucket_name}")

    def upload_file(self, local_file_path: str, s3_object_key: str) -> bool:
        """
        Uploads a file from the local filesystem to S3.
        """
        if not os.path.exists(local_file_path):
            logger.error(f"Local file not found: {local_file_path}")
            return False

        try:
            logger.info(f"Starting upload of '{local_file_path}' to '{self.bucket_name}/{s3_object_key}'")
            
            self.s3.upload_file(local_file_path, self.bucket_name, s3_object_key)
            
            logger.info(f"Successfully uploaded {s3_object_key}")
            return True
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Ensure .env or environment variables are loaded.")
            return False
        except ClientError as e:
            logger.error(f"S3 Client Error uploading {s3_object_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during upload: {e}")
            return False

# --- Example Execution (Demonstration) ---
if __name__ == "__main__":
    # NOTE: In your local environment, you would use `from dotenv import load_dotenv; load_dotenv()` here.
    
    logger.info("--- Running S3 Uploader Demonstration ---")
    
    config = load_config()
    if not config:
        sys.exit(1)
        
    S3_BUCKET = config['s3_data_paths']['bucket_name']
    
    # 1. Create a dummy file to upload
    DUMMY_FILE = "dummy_upload_s3.txt"
    with open(DUMMY_FILE, 'w') as f:
        f.write(f"Test upload at {datetime.now()}")

    # 2. Initialize the uploader
    uploader = S3Uploader(bucket_name=S3_BUCKET)

    # 3. Define the S3 destination path
    S3_DEST_KEY = f"raw_data/test_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # 4. Perform the upload
    success = uploader.upload_file(
        local_file_path=DUMMY_FILE,
        s3_object_key=S3_DEST_KEY
    )

    # 5. Clean up the dummy file
    os.remove(DUMMY_FILE)
    
    logger.info(f"Upload attempt result: {success}")
    # utils/s3_uploader.py

# ... (rest of the file remains the same)

# --- Example Execution (Demonstration) ---
if __name__ == "__main__":
    # --- ADD THIS IMPORT ---
    from dotenv import load_dotenv 

    logger.info("--- Running S3 Uploader Demonstration ---")
    
    # 1. LOAD THE .ENV FILE
    # This must be the first step to make keys available via os.environ
    load_dotenv() 

    config = load_config()
    if not config:
        sys.exit(1)
        
    S3_BUCKET = config['s3_data_paths']['bucket_name']
    
    # Credentials will now be automatically picked up by boto3.Session, 
    # but we can explicitly pass them for robustness if needed:
    aws_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1") # Fallback to a default region

    # 2. Create a dummy file to upload
    DUMMY_FILE = "dummy_upload_s3.txt"
    with open(DUMMY_FILE, 'w') as f:
        f.write(f"Test upload at {datetime.now()}")

    # 3. Initialize the uploader
    uploader = S3Uploader(
        bucket_name=S3_BUCKET,
        region_name=aws_region
        # Note: Boto3 will now find credentials via load_dotenv
    )

    # 4. Define the S3 destination path
    S3_DEST_KEY = f"raw_data/test_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # 5. Perform the upload
    success = uploader.upload_file(
        local_file_path=DUMMY_FILE,
        s3_object_key=S3_DEST_KEY
    )

    # 6. Clean up the dummy file
    os.remove(DUMMY_FILE)
    
    logger.info(f"Upload attempt result: {success}")