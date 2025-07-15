import logging
from threading import Lock

class Logger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.logger = logging.getLogger("FusionAIHubLogger")
        self.logger.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

logger = Logger()

# Usage example:
# logger = Logger().get_logger()
# logger.info("This is a singleton logger.")