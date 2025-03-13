#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the Pet Dialogue application.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Dict, List, Any, Optional, Union

def setup_logging():
    """Configure logging for the application."""
    log_level = os.getenv("LOG_LEVEL", "DEBUG")
    
    # Remove default logger
    logger.remove()
    
    # Add stdout logger with specified log level
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"pet_dialogue_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        level=log_level
    )
    
    logger.info(f"Logging initialized at level {log_level}")

def decode_image(image_data: bytes) -> np.ndarray:
    """
    Decode image data from bytes to OpenCV format.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Image as numpy array in BGR format
    """
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image(image: np.ndarray, format: str = ".jpg") -> bytes:
    """
    Encode OpenCV image to bytes.
    
    Args:
        image: OpenCV image (numpy array)
        format: Image format (".jpg" or ".png")
        
    Returns:
        Encoded image bytes
    """
    success, buffer = cv2.imencode(format, image)
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()

def load_json_data(file_path: Union[str, Path]) -> Dict:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data as dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data: Dict, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent 