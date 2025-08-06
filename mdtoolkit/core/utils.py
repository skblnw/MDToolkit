"""General utilities for the MD toolkit."""

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Union, Callable
import yaml
import numpy as np


def setup_logger(
    name: str = "md-toolkit",
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Parameters
    ----------
    name : str, default "md-toolkit"
        Logger name
    level : str or int, default "INFO"
        Logging level
    log_file : str or Path, optional
        Path to log file
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Parameters
    ----------
    func : callable
        Function to time
        
    Returns
    -------
    callable
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(
            f"{func.__name__} took {end_time - start_time:.3f} seconds"
        )
        
        return result
    
    return wrapper


def load_config(config_file: Union[str, Path]) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_file : str or Path
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict, config_file: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_file : str or Path
        Path to save configuration
    """
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def chunk_trajectory(
    n_frames: int,
    chunk_size: int = 1000
) -> list:
    """
    Split trajectory into chunks for processing.
    
    Parameters
    ----------
    n_frames : int
        Total number of frames
    chunk_size : int, default 1000
        Size of each chunk
        
    Returns
    -------
    list of tuples
        List of (start, stop) indices for each chunk
    """
    chunks = []
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        chunks.append((start, stop))
    
    return chunks


def calculate_block_average(
    data: np.ndarray,
    block_size: int
) -> tuple:
    """
    Calculate block averages for error estimation.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    block_size : int
        Size of blocks
        
    Returns
    -------
    tuple
        (block_averages, block_std, block_error)
    """
    n_blocks = len(data) // block_size
    
    if n_blocks < 2:
        raise ValueError(
            f"Not enough data for block size {block_size}. "
            f"Need at least {2 * block_size} points."
        )
    
    # Truncate data to fit blocks
    truncated_data = data[:n_blocks * block_size]
    
    # Reshape into blocks
    blocks = truncated_data.reshape(n_blocks, block_size)
    
    # Calculate block averages
    block_averages = np.mean(blocks, axis=1)
    
    # Calculate statistics
    block_mean = np.mean(block_averages)
    block_std = np.std(block_averages, ddof=1)
    block_error = block_std / np.sqrt(n_blocks)
    
    return block_averages, block_std, block_error


def running_average(
    data: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Calculate running average of data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    window_size : int
        Window size for averaging
        
    Returns
    -------
    np.ndarray
        Running average
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    if window_size == 1:
        return data
    
    # Use convolution for efficiency
    kernel = np.ones(window_size) / window_size
    
    # Handle edge effects
    mode = 'valid' if len(data) > window_size else 'same'
    
    return np.convolve(data, kernel, mode=mode)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"