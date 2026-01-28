"""
Unified training pipeline for all multi-document summarization models.
Handles training, validation, early stopping, and checkpoint management.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json

from config import TRAINING_CONFIG, CHECKPOINT_ROOT
from evaluation.metrics import EvaluationEngine
from utils.utils import setup_logging, batch_iterator, ProgressTracker

logger = setup_logging(__name__)


class UnifiedTrainer:
    """Unified training loop for all models."""
    
    def __init__(self, model, model_id: str, device: str = "cuda"):
        self.model = model
        self.model_id = model_id
        self.device = device
        self.evaluator = EvaluationEngine(device=device)
        
        self.checkpoint_dir = CHECKPOINT_ROOT / model_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_bertscore": [],
            "epochs": [],
        }
    
    def prepare_batch_for_training(self, cluster_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training.
        Must be implemented by each model wrapper.
        """
        # This should be implemented in model-specific training wrappers
        raise NotImplementedError("Subclass must implement prepare_batch_for_training")
    
    def train_epoch(
        self,
        train_clusters: List[Dict],
        batch_size: int = 4
    ) -> float:
        """Train for one epoch."""
        self.model.set_train_mode()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(total=len(train_clusters), desc=f"Epoch {self.current_epoch+1} - Train")
        
        # Group clusters into batches
        for batch_clusters in batch_iterator(train_clusters, batch_size):
            batch_loss = 0.0
            
            for cluster_data in batch_clusters:
                try:
                    batch = self.prepare_batch_for_training(cluster_data)
                    metrics = self.model.train_step(batch)
                    batch_loss += metrics.get("loss", 0.0)
                except Exception as e:
                    logger.warning(f"Error in batch for model {self.model_id}: {str(e)[:100]}")
                    continue
            
            if batch_loss > 0:
                avg_batch_loss = batch_loss / max(len(batch_clusters), 1)
                total_loss += avg_batch_loss
                num_batches += 1
            
            pbar.update(len(batch_clusters))
        
        pbar.close()
        
        epoch_loss = total_loss / max(num_batches, 1)
        self.training_history["train_loss"].append(epoch_loss)
        
        logger.info(f"{self.model_id} - Epoch {self.current_epoch+1} - Train Loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def validate_epoch(
        self,
        val_clusters: List[Dict],
        batch_size: int = 4
    ) -> Dict[str, float]:
        """Validation epoch."""
        self.model.set_eval_mode()
        total_loss = 0.0
        num_batches = 0
        
        all_generated = []
        all_references = []
        
        pbar = tqdm(total=len(val_clusters), desc=f"Epoch {self.current_epoch+1} - Val")
        
        with torch.no_grad():
            for batch_clusters in batch_iterator(val_clusters, batch_size):
                for cluster_data in batch_clusters:
                    try:
                        # Get documents and reference
                        documents = [a["article_text"] for a in cluster_data["articles"]]
                        reference = cluster_data["reference_summary"]
                        
                        # Generate summary
                        generated = self.model.generate_summary(documents)
                        
                        all_generated.append(generated)
                        all_references.append(reference)
                        
                    except Exception as e:
                        logger.warning(f"Error in validation: {str(e)[:100]}")
                        continue
                
                pbar.update(len(batch_clusters))
        
        pbar.close()
        
        # Compute metrics
        val_metrics = {}
        if all_generated and all_references:
            try:
                # Simple BERTScore computation
                from bert_score import score as bert_score
                _, _, F1 = bert_score(
                    all_generated,
                    all_references,
                    lang="en",
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=self.device,
                    batch_size=8
                )
                val_metrics["bertscore_f1"] = float(F1.mean())
            except Exception as e:
                logger.warning(f"Error computing BERTScore: {e}")
                val_metrics["bertscore_f1"] = 0.0
        
        self.training_history["val_bertscore"].append(val_metrics.get("bertscore_f1", 0.0))
        
        logger.info(f"{self.model_id} - Epoch {self.current_epoch+1} - Val BERTScore: {val_metrics.get('bertscore_f1', 0.0):.4f}")
        
        return val_metrics
    
    def train(
        self,
        train_clusters: List[Dict],
        val_clusters: List[Dict],
        num_epochs: int = TRAINING_CONFIG["num_epochs"],
        batch_size: int = TRAINING_CONFIG["batch_size"],
        learning_rate: float = TRAINING_CONFIG["learning_rate"],
        early_stopping_patience: int = TRAINING_CONFIG["early_stopping_patience"],
    ) -> Dict:
        """Full training loop."""
        logger.info("="*80)
        logger.info(f"STARTING TRAINING FOR {self.model_id}")
        logger.info("="*80)
        logger.info(f"Train clusters: {len(train_clusters)}, Val clusters: {len(val_clusters)}")
        logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Setup optimizer and scheduler
        num_training_steps = (len(train_clusters) // batch_size + 1) * num_epochs
        self.model.setup_optimizer(learning_rate)
        self.model.setup_scheduler(num_training_steps)
        
        self.patience_counter = 0
        self.best_metric = -float('inf')
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_clusters, batch_size)
            
            # Validate
            val_metrics = self.validate_epoch(val_clusters, batch_size)
            
            # Early stopping
            current_metric = val_metrics.get("bertscore_f1", -float('inf'))
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = self.checkpoint_dir / f"best_model.pt"
                self.model.save_checkpoint(
                    checkpoint_path,
                    epoch,
                    {"train_loss": train_loss, **val_metrics}
                )
                logger.info(f"New best model saved: {checkpoint_path}")
            else:
                self.patience_counter += 1
                logger.info(f"Patience: {self.patience_counter}/{early_stopping_patience}")
                
                if self.patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("="*80)
        logger.info(f"TRAINING COMPLETE FOR {self.model_id}")
        logger.info(f"Best BERTScore: {self.best_metric:.4f}")
        logger.info("="*80)
        
        return {
            "model_id": self.model_id,
            "epochs_trained": self.current_epoch + 1,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
        }
    
    def save_training_logs(self) -> None:
        """Save training history."""
        log_path = self.checkpoint_dir / "training_logs.json"
        
        logs = {
            "model_id": self.model_id,
            "training_history": self.training_history,
            "best_metric": self.best_metric,
        }
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Training logs saved: {log_path}")


# ==================== MODEL-SPECIFIC TRAINING WRAPPERS ====================

class PEGASUSTrainer(UnifiedTrainer):
    """Training wrapper for PEGASUS."""
    
    def prepare_batch_for_training(self, cluster_data: Dict) -> Dict[str, torch.Tensor]:
        """Prepare batch for PEGASUS training."""
        from models.pegasus_model import PEGASUSTrainingWrapper
        
        wrapper = PEGASUSTrainingWrapper(device=self.device)
        return wrapper.prepare_batch(cluster_data)


class LEDTrainer(UnifiedTrainer):
    """Training wrapper for LED."""
    
    def prepare_batch_for_training(self, cluster_data: Dict) -> Dict[str, torch.Tensor]:
        """Prepare batch for LED training."""
        from models.led_model import LEDSummarizer
        from utils.utils import combine_documents
        
        documents = [article["article_text"] for article in cluster_data["articles"]]
        reference_summary = cluster_data["reference_summary"]
        
        # Tokenize inputs (LED can handle longer sequences)
        inputs = self.model.tokenizer(
            combine_documents(documents, method="separated"),
            max_length=16384,
            truncation=True,
            return_tensors="pt",
            padding="max_length"
        )
        
        # Add global attention mask
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        inputs["global_attention_mask"] = global_attention_mask
        
        # Tokenize targets
        with self.model.tokenizer.as_target_tokenizer():
            labels = self.model.tokenizer(
                reference_summary,
                max_length=256,
                truncation=True,
                return_tensors="pt",
                padding="max_length"
            )
        
        batch = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "global_attention_mask": inputs["global_attention_mask"].to(self.device),
            "labels": labels["input_ids"].to(self.device),
        }
        
        # Mask padding tokens
        batch["labels"][batch["labels"] == self.model.tokenizer.pad_token_id] = -100
        
        return batch


class AIMSTrainer(UnifiedTrainer):
    """Training wrapper for AIMS (delegates to PEGASUS)."""
    
    def prepare_batch_for_training(self, cluster_data: Dict) -> Dict[str, torch.Tensor]:
        """Prepare batch for AIMS training."""
        from models.pegasus_model import PEGASUSTrainingWrapper
        
        wrapper = PEGASUSTrainingWrapper(device=self.device)
        return wrapper.prepare_batch(cluster_data)
