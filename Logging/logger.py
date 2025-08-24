import os
import sys
import logging
from datetime import datetime


logging_format = "[%(asctime)s] - %(lineno)d %(name)s - %(levelname)s - %(module)s: - %(message)s"
log_dir = os.path.join(os.getcwd(),"Artifacts/Logs")
os.makedirs(log_dir,exist_ok=True)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_filepath = os.path.join(log_dir,LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    handlers=[
        logging.FileHandler(log_filepath), # putting log message in the logs file.
        logging.StreamHandler(sys.stdout)  # printing log message on the console.
    ]
)

logger = logging.getLogger("logger")