"""
Comprehensive evaluation metrics for multi-document summarization.
Includes ROUGE, BERTScore, redundancy, omission, hallucination, faithfulness.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import spacy
from collections import Counter
import re

from utils.utils import setup_logging

logger = setup_logging(__name__)


class MetricsComputer:
    """Compute all evaluation metrics."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
        
        # Load spaCy NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
    
    # ==================== ROUGE METRICS ====================
    def compute_rouge(self, reference: str, generated: str) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    
    # ==================== BERTSCORE ====================
    def compute_bertscore(
        self,
        references: List[str],
        generated: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute BERTScore for batch of summaries.
        
        Returns: (precision, recall, f1) as numpy arrays
        """
        if not references or not generated:
            return np.array([]), np.array([]), np.array([])
        
        P, R, F1 = bert_score(
            generated,
            references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            device=self.device,
            batch_size=16,
            verbose=False
        )
        
        return P.cpu().numpy(), R.cpu().numpy(), F1.cpu().numpy()
    
    # ==================== REDUNDANCY RATE ====================
    def compute_redundancy_rate(self, text: str) -> float:
        """
        Compute redundancy rate based on:
        1. Repeated n-grams (3-grams)
        2. Sentence similarity
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        if len(sentences) <= 1:
            return 0.0
        
        # 1. N-gram redundancy
        all_ngrams = []
        for sent in sentences:
            ngrams = self._extract_ngrams(sent, n=3)
            all_ngrams.extend(ngrams)
        
        if len(all_ngrams) == 0:
            return 0.0
        
        ngram_counts = Counter(all_ngrams)
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        ngram_redundancy = repeated_ngrams / len(all_ngrams) if len(all_ngrams) > 0 else 0
        
        # 2. Sentence similarity redundancy
        similarity_scores = []
        for i in range(len(sentences) - 1):
            for j in range(i + 1, len(sentences)):
                if sentences[i].strip() and sentences[j].strip():
                    sim = self._sentence_similarity(sentences[i], sentences[j])
                    if sim > 0.7:
                        similarity_scores.append(sim)
        
        sentence_redundancy = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Combined (weighted)
        redundancy_score = 0.6 * ngram_redundancy + 0.4 * sentence_redundancy
        
        return min(redundancy_score, 1.0)
    
    def _extract_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract n-grams."""
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Compute sentence similarity using simple token overlap."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    # ==================== OMISSION RATE ====================
    def compute_omission_rate(self, reference: str, generated: str) -> float:
        """
        Omission rate = fraction of entities/facts from reference not in generated.
        """
        ref_entities = self._extract_entities(reference)
        gen_entities = self._extract_entities(generated)
        
        total_ref_entities = sum(len(ents) for ents in ref_entities.values())
        
        if total_ref_entities == 0:
            return 0.0
        
        missing_count = 0
        for ent_type, ref_ents in ref_entities.items():
            gen_ents = gen_entities.get(ent_type, set())
            missing = ref_ents - gen_ents
            missing_count += len(missing)
        
        omission_rate = missing_count / total_ref_entities
        
        return min(omission_rate, 1.0)
    
    def _extract_entities(self, text: str) -> Dict[str, set]:
        """Extract named entities."""
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            ent_type = ent.label_
            if ent_type not in entities:
                entities[ent_type] = set()
            entities[ent_type].add(ent.text.lower())
        
        return entities
    
    # ==================== HALLUCINATION RATE ====================
    def compute_hallucination_rate(
        self,
        source_docs: List[str],
        generated: str,
        reference: str
    ) -> float:
        """
        Hallucination rate = fraction of entities in generated not in source + reference.
        """
        # Combine source + reference
        combined_source = ' '.join(source_docs + [reference])
        source_entities = self._extract_entities(combined_source)
        gen_entities = self._extract_entities(generated)
        
        total_gen_entities = sum(len(ents) for ents in gen_entities.values())
        
        if total_gen_entities == 0:
            return 0.0
        
        hallucinated_count = 0
        for ent_type, gen_ents in gen_entities.items():
            source_ents = source_entities.get(ent_type, set())
            hallucinated = gen_ents - source_ents
            hallucinated_count += len(hallucinated)
        
        hallucination_rate = hallucinated_count / total_gen_entities
        
        return min(hallucination_rate, 1.0)
    
    # ==================== FAITHFULNESS SCORE ====================
    def compute_faithfulness_score(
        self,
        source_docs: List[str],
        generated: str
    ) -> float:
        """
        Faithfulness = 1 - hallucination_rate
        (Fraction of facts that are faithful to source)
        """
        combined_source = ' '.join(source_docs)
        hallucination = self.compute_hallucination_rate(
            source_docs=[combined_source],
            generated=generated,
            reference=""
        )
        
        return 1.0 - hallucination
    
    # ==================== COMPRESSION RATIO ====================
    def compute_compression_ratio(
        self,
        source_docs: List[str],
        generated: str
    ) -> float:
        """Compression ratio = generated_words / source_words."""
        source_words = sum(len(doc.split()) for doc in source_docs)
        gen_words = len(generated.split())
        
        if source_words == 0:
            return 0.0
        
        return gen_words / source_words


class EvaluationEngine:
    """Unified evaluation engine for all metrics."""
    
    def __init__(self, device: str = "cpu"):
        self.metrics_computer = MetricsComputer(device=device)
    
    def evaluate_single(
        self,
        source_docs: List[str],
        reference: str,
        generated: str
    ) -> Dict[str, float]:
        """
        Evaluate a single summary.
        
        Returns: Dictionary with all metrics
        """
        results = {}
        
        # ROUGE
        rouge_scores = self.metrics_computer.compute_rouge(reference, generated)
        results.update(rouge_scores)
        
        # Redundancy
        results["redundancy_rate"] = self.metrics_computer.compute_redundancy_rate(generated)
        
        # Omission
        results["omission_rate"] = self.metrics_computer.compute_omission_rate(reference, generated)
        
        # Hallucination
        results["hallucination_rate"] = self.metrics_computer.compute_hallucination_rate(
            source_docs, generated, reference
        )
        
        # Faithfulness
        results["faithfulness"] = self.metrics_computer.compute_faithfulness_score(
            source_docs, generated
        )
        
        # Compression
        results["compression_ratio"] = self.metrics_computer.compute_compression_ratio(
            source_docs, generated
        )
        
        return results
    
    def evaluate_batch(
        self,
        source_docs_list: List[List[str]],
        references: List[str],
        generated_list: List[str]
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """
        Evaluate batch of summaries.
        
        Returns:
            (per_sample_metrics, aggregate_metrics)
        """
        per_sample = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bertscore_f1": [],
            "redundancy_rate": [],
            "omission_rate": [],
            "hallucination_rate": [],
            "faithfulness": [],
            "compression_ratio": [],
        }
        
        # Individual metrics
        for source_docs, reference, generated in zip(source_docs_list, references, generated_list):
            metrics = self.evaluate_single(source_docs, reference, generated)
            for key, value in metrics.items():
                per_sample[key].append(value)
        
        # BERTScore (batch)
        P, R, F1 = self.metrics_computer.compute_bertscore(references, generated_list)
        per_sample["bertscore_f1"] = F1.tolist()
        
        # Aggregate
        aggregate = {}
        for key, values in per_sample.items():
            if values:
                aggregate[f"{key}_mean"] = np.mean(values)
                aggregate[f"{key}_std"] = np.std(values)
                aggregate[f"{key}_min"] = np.min(values)
                aggregate[f"{key}_max"] = np.max(values)
        
        return per_sample, aggregate


if __name__ == "__main__":
    engine = EvaluationEngine(device="cpu")
    
    # Example
    source = ["Article 1 text here", "Article 2 text here"]
    reference = "Reference summary"
    generated = "Generated summary"
    
    metrics = engine.evaluate_single(source, reference, generated)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
