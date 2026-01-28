"""
Multi-Document Summarization Benchmarking Pipeline
Complete framework for training and evaluating 11 summarization models
"""

__version__ = "1.0.0"
__author__ = "ML Research Team"

from config import TRAINING_CONFIG, DATASET_CONFIG, MODELS_TO_BENCHMARK
from run_benchmarking import BenchmarkingOrchestrator

__all__ = [
    "BenchmarkingOrchestrator",
    "TRAINING_CONFIG",
    "DATASET_CONFIG",
    "MODELS_TO_BENCHMARK",
]
