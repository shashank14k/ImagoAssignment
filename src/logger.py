import logging
import os
import sys

class MainLogger:
    def __init__(self, log_file=None, log_level=logging.INFO):
        """
        Initialize the logger.
        :param log_file: (str) Path to log file. If None, logs to console.
        :param log_level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger("MainLogger")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Prevent duplicate logs

        # Remove existing handlers (useful if re-initializing)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if log_file:
            # Log to file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            # Log to console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

log_file_name = os.environ.get("log_file_path", None)
logger = MainLogger(log_file_name)
