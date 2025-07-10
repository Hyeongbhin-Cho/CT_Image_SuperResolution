# utils/logger.py

import os
import logging
from datetime import datetime

def get_logger(log_dir: str, name: str = 'train') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    # Log filename
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{time_str}.log')
    
    # Prevent duplication
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    

    return logger