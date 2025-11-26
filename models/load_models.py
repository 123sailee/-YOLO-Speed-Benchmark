"""Load and initialize YOLO models."""

from ultralytics import YOLO
from pathlib import Path


def load_yolo_model(model_name: str, device: str = None):
    """
    Load a YOLO model by name.
    
    Args:
        model_name (str): Model identifier (e.g., 'yolov5n', 'yolov8s', 'yolov11m')
        device (str): Device to load model on ('cpu', 'cuda:0', etc.)
    
    Returns:
        YOLO: Loaded model instance
    """
    try:
        model = YOLO(model_name)
        if device:
            model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def get_available_models():
    """Return list of available YOLO model variants."""
    return {
        'yolov5': ['yolov5n', 'yolov5s', 'yolov5m'],
        'yolov8': ['yolov8n', 'yolov8s', 'yolov8m'],
        'yolov11': ['yolov11n', 'yolov11s', 'yolov11m'],
    }
