"""
Configuration for multi-model benchmarking pipeline.
Defines all hyperparameters, model configs, and pipeline settings.
"""

import os
from pathlib import Path

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARKING_ROOT = Path(__file__).parent
DATA_ROOT = BENCHMARKING_ROOT / "data"
CHECKPOINT_ROOT = BENCHMARKING_ROOT / "checkpoints"
RESULTS_ROOT = BENCHMARKING_ROOT / "results"

# Create directories if they don't exist
for path in [DATA_ROOT, CHECKPOINT_ROOT, RESULTS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

# ==================== DATASET CONFIG ====================
DATASET_CONFIG = {
    "name": "newssumm",
    "raw_path": PROJECT_ROOT / "data" / "processed" / "newssumm_clean.csv",
    "processed_path": DATA_ROOT / "clusters.json",
    "train_clusters_path": DATA_ROOT / "train_clusters.json",
    "val_clusters_path": DATA_ROOT / "val_clusters.json",
    "test_clusters_path": DATA_ROOT / "test_clusters.json",
    "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    "min_cluster_size": 2,  # Minimum articles per cluster
    "max_cluster_size": 20,  # Maximum articles per cluster
    "temporal_window_days": 1,  # For clustering
    "similarity_threshold": 0.60,  # Semantic similarity
}

# ==================== TRAINING CONFIG ====================
TRAINING_CONFIG = {
    "device": "cuda",  # "cuda" or "cpu"
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "max_seq_length": 1024,
    "max_target_length": 256,
    "early_stopping_patience": 3,
    "early_stopping_metric": "bertscore_f1",  # Metric to monitor
    "save_best_only": True,
    "logging_steps": 50,
    "eval_steps": 500,
    "seed": 42,
}

# ==================== INFERENCE CONFIG ====================
INFERENCE_CONFIG = {
    "max_length": 256,
    "min_length": 30,
    "num_beams": 4,
    "repetition_penalty": 2.0,
    "length_penalty": 1.0,
    "early_stopping": True,
    "temperature": 1.0,
    "do_sample": False,
}

# ==================== MODEL CONFIGS ====================
MODELS_TO_BENCHMARK = {
    "pegasus": {
        "model_name": "google/pegasus-arxiv",
        "type": "seq2seq",
        "trainable": True,
        "max_input": 1024,
        "max_target": 256,
    },
    "led": {
        "model_name": "allenai/led-base-16384",
        "type": "seq2seq",
        "trainable": True,
        "max_input": 16384,
        "max_target": 256,
    },
    "bigbird_pegasus": {
        "model_name": "google/bigbird-pegasus-large-arxiv",
        "type": "seq2seq",
        "trainable": True,
        "max_input": 4096,
        "max_target": 256,
    },
    "primera": {
        "model_name": "allenai/primera",
        "type": "seq2seq",
        "trainable": True,
        "max_input": 4096,
        "max_target": 256,
    },
    "longt5": {
        "model_name": "google/long-t5-tglobal-base",
        "type": "seq2seq",
        "trainable": True,
        "max_input": 16384,
        "max_target": 256,
    },
    "graphsum": {
        "type": "graph_based",
        "trainable": True,
        "graph_type": "inter_document",
        "max_input": 1024,
        "max_target": 256,
    },
    "llm_instruction": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "llm",
        "trainable": False,  # Using instruction following
        "max_input": 4096,
        "max_target": 256,
    },
    "factuality_aware": {
        "type": "factuality_verification",
        "generator": "google/pegasus-arxiv",
        "verifier": "sentence-transformers/all-MiniLM-L6-v2",
        "trainable": True,
        "max_input": 1024,
        "max_target": 256,
    },
    "event_aware": {
        "type": "event_clustering_based",
        "backbone": "google/pegasus-arxiv",
        "trainable": True,
        "max_input": 1024,
        "max_target": 256,
    },
    "benchmark_llm": {
        "model_name": "gpt2",  # Fallback to open model; user can replace with API
        "type": "llm",
        "trainable": False,
        "max_input": 1024,
        "max_target": 256,
    },
    "aims": {
        "type": "importance_aware",
        "backbone": "google/pegasus-arxiv",
        "embedding_model": "all-MiniLM-L6-v2",
        "trainable": True,
        "max_input": 1024,
        "max_target": 256,
    },
}

# ==================== EVALUATION METRICS ====================
METRICS_CONFIG = {
    "rouge": {
        "types": ["rouge1", "rouge2", "rougeL"],
        "use_stemmer": True,
    },
    "bertscore": {
        "model_type": "microsoft/deberta-xlarge-mnli",
        "lang": "en",
    },
    "redundancy": {
        "ngram_size": 3,
        "similarity_threshold": 0.7,
    },
    "omission": {
        "method": "ner_based",  # Using named entity extraction
    },
    "hallucination": {
        "method": "entailment_based",  # Using NLI model
    },
    "faithfulness": {
        "nli_model": "microsoft/deberta-base",
    },
}

# ==================== STATISTICAL TESTING ====================
STATS_CONFIG = {
    "bootstrap_samples": 10000,
    "confidence_level": 0.95,
    "significance_level": 0.05,
    "test_type": "paired_bootstrap",  # For paired comparisons (AIMS vs others)
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BENCHMARKING_ROOT / "benchmarking.log",
}

# ==================== REPRODUCIBILITY ====================
REPRODUCIBILITY = {
    "seed": 42,
    "deterministic": True,
    "benchmark": False,  # Set to True for reproducibility (slower)
}
