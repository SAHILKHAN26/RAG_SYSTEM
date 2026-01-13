import logging
import sys
import os
from datetime import datetime

class AppLogger:
    def __init__(self, name: str = 'app', log_file: str = 'app.log'):

        current_date = datetime.now().strftime("%d-%m-%Y")
        log_file = f"app-{current_date}.log"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        log_folder_path = "logs"
        isExist = os.path.exists(log_folder_path)
        if not isExist:
            os.makedirs(log_folder_path)
            print("log folder created")

        file_handler = logging.FileHandler(os.path.abspath("logs/"+log_file))
        file_handler.setFormatter(formatter)

        self.logger.handlers = [stream_handler, file_handler]

    def get_logger(self):
        return self.logger
    
            
    
# Singleton instance of the logger
app_logger = AppLogger().get_logger()