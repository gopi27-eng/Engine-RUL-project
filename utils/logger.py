import logging
import os
from datetime import datetime

# Define log file name and path
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILEPATH,
    level=logging.INFO,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("RULLogger")
logger.info(f"Log file initialized at: {LOG_FILEPATH}")

# Add console handler for real-time visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[ %(levelname)s ] %(message)s'))
logger.addHandler(console_handler)

# Set logging level (can be DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)