import logging
import sys
from typing import Optional
from pathlib import Path


class Logger:
    """Centralized logging utility for the knowledge graph pipeline."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._logger = logging.getLogger("kg_pipeline")
            self._logger.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self._logger.addHandler(console_handler)
    
    @classmethod
    def configure(cls, level: str = "INFO", log_file: Optional[Path] = None):
        """Configure the logger with specific settings."""
        logger = cls()
        
        # Set log level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level.upper() in level_map:
            logger._logger.setLevel(level_map[level.upper()])
        
        # Add file handler if log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger._logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls):
        """Get the configured logger instance."""
        return cls()._logger
    
    @classmethod
    def debug(cls, message: str):
        """Log debug message."""
        cls.get_logger().debug(message)
    
    @classmethod
    def info(cls, message: str):
        """Log info message."""
        cls.get_logger().info(message)
    
    @classmethod
    def warning(cls, message: str):
        """Log warning message."""
        cls.get_logger().warning(message)
    
    @classmethod
    def error(cls, message: str):
        """Log error message."""
        cls.get_logger().error(message)
    
    @classmethod
    def critical(cls, message: str):
        """Log critical message."""
        cls.get_logger().critical(message) 