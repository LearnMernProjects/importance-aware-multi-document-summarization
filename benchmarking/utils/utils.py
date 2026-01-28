"""
Utility functions for logging, data handling, and reproducibility.
"""

import logging
import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from config import LOGGING_CONFIG, REPRODUCIBILITY

# ==================== LOGGING SETUP ====================
def setup_logging(name: str) -> logging.Logger:
    """Configure logging with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG["level"])
    
    # Formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # File handler
    fh = logging.FileHandler(LOGGING_CONFIG["log_file"])
    fh.setLevel(LOGGING_CONFIG["level"])
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(LOGGING_CONFIG["level"])
    ch.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

# ==================== REPRODUCIBILITY ====================
def set_seed(seed: int = None) -> None:
    """Set seeds for reproducibility across libraries."""
    if seed is None:
        seed = REPRODUCIBILITY["seed"]
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if REPRODUCIBILITY["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Seed set to {seed} for reproducibility")

# ==================== JSON UTILITIES ====================
def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: Path) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==================== TEXT UTILITIES ====================
def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length (by characters)."""
    if len(text) > max_length:
        return text[:max_length]
    return text

def combine_documents(docs: List[str], method: str = "concatenate") -> str:
    """
    Combine multiple documents into a single input.
    
    Args:
        docs: List of document texts
        method: "concatenate", "numbered", or "separated"
    
    Returns:
        Combined text
    """
    if method == "concatenate":
        return "\n\n".join(docs)
    elif method == "numbered":
        return "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(docs)])
    elif method == "separated":
        return "\n\n[DOCUMENT BOUNDARY]\n\n".join(docs)
    else:
        return "\n\n".join(docs)

# ==================== CHECKPOINT UTILITIES ====================
def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    model_name: str = "model"
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict() if hasattr(model, 'state_dict') else None,
        "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    torch.save(checkpoint, path)

def load_checkpoint(path: Path, model: torch.nn.Module, device: str = "cpu"):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    if checkpoint["model_state_dict"] is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    return checkpoint

# ==================== METRICS UTILITIES ====================
def compute_average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute average and std of metrics across samples."""
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].keys()
    averages = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if values:
            averages[f"{key}_mean"] = np.mean(values)
            averages[f"{key}_std"] = np.std(values)
    
    return averages

# ==================== BATCH UTILITIES ====================
def batch_iterator(data: List[Any], batch_size: int):
    """Iterate over data in batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# ==================== PROGRESS TRACKING ====================
class ProgressTracker:
    """Track training progress."""
    
    def __init__(self, total_steps: int, log_interval: int = 10):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.metrics_buffer = {}
    
    def update(self, step: int, metrics: Dict[str, float]) -> None:
        """Update progress."""
        self.current_step = step
        for key, value in metrics.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            self.metrics_buffer[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics since last log."""
        avg_metrics = {}
        for key, values in self.metrics_buffer.items():
            if values:
                avg_metrics[key] = np.mean(values)
        self.metrics_buffer = {}
        return avg_metrics
    
    def should_log(self) -> bool:
        """Check if should log."""
        return self.current_step % self.log_interval == 0

if __name__ == "__main__":
    set_seed()
    logger = setup_logging(__name__)
    logger.info("Utilities module loaded successfully")
