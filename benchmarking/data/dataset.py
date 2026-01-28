"""
Data pipeline for loading, clustering, and splitting NewsSumm dataset.
Creates event-level multi-document clusters with train/val/test splits.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

from config import DATASET_CONFIG, TRAINING_CONFIG
from utils.utils import setup_logging, save_json, load_json, set_seed

logger = setup_logging(__name__)

class NewsSummClusterer:
    """Cluster NewsSumm articles by real-world events."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.60,
        temporal_window_days: int = 1,
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.temporal_window_days = temporal_window_days
        logger.info(f"Initialized clusterer with model: {embedding_model}")
    
    def load_dataset(self, csv_path: Path) -> pd.DataFrame:
        """Load and preprocess NewsSumm dataset."""
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Parse dates
        df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        
        # Remove invalid rows
        initial_len = len(df)
        df = df.dropna(subset=['article_text', 'published_date'])
        df = df[df['article_text'].str.len() > 50]
        
        logger.info(f"Loaded {len(df)} articles (removed {initial_len - len(df)} invalid)")
        logger.info(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
        
        return df.sort_values('published_date').reset_index(drop=True)
    
    def cluster_articles(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Cluster articles into event-level groups.
        Returns: {cluster_id: [article_indices]}
        """
        logger.info("Starting event-level clustering...")
        
        # Create combined text for embedding
        df['combined_text'] = (
            df['headline'].fillna('') + ' ' + 
            df['article_text'].fillna('')
        ).str.strip()
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(df)} articles...")
        embeddings = self.embedding_model.encode(
            df['combined_text'].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        # Incremental clustering
        clusters = {}
        article_to_cluster = {}
        cluster_counter = 0
        
        for i in range(len(df)):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(df)} articles, {cluster_counter} clusters formed")
            
            if i in article_to_cluster:
                continue  # Already assigned
            
            # Find candidates with temporal and category constraints
            current_date = df.loc[i, 'published_date']
            current_category = df.loc[i, 'news_category']
            
            candidates = []
            for j in range(i + 1, len(df)):
                if j in article_to_cluster:
                    continue
                
                # Temporal constraint
                date_diff = (df.loc[j, 'published_date'] - current_date).days
                if abs(date_diff) > self.temporal_window_days:
                    break  # Since sorted by date, no more candidates
                
                # Category constraint
                if df.loc[j, 'news_category'] != current_category:
                    continue
                
                candidates.append(j)
            
            # Find most similar candidate
            best_candidate = None
            best_similarity = -1
            
            if candidates:
                similarities = cosine_similarity(
                    embeddings[i:i+1],
                    embeddings[candidates]
                )[0]
                
                max_sim_idx = np.argmax(similarities)
                if similarities[max_sim_idx] > self.similarity_threshold:
                    best_candidate = candidates[max_sim_idx]
                    best_similarity = similarities[max_sim_idx]
            
            # Assign to cluster
            if best_candidate is not None and best_candidate in article_to_cluster:
                cluster_id = article_to_cluster[best_candidate]
                clusters[cluster_id].append(i)
                article_to_cluster[i] = cluster_id
                logger.debug(f"Article {i} assigned to cluster {cluster_id} (sim={best_similarity:.3f})")
            else:
                # Create new cluster
                clusters[cluster_counter] = [i]
                article_to_cluster[i] = cluster_counter
                cluster_counter += 1
        
        logger.info(f"Clustering complete: {cluster_counter} clusters, {len(article_to_cluster)} articles")
        
        # Validate clusters
        valid_clusters = {
            cid: articles for cid, articles in clusters.items()
            if DATASET_CONFIG["min_cluster_size"] <= len(articles) <= DATASET_CONFIG["max_cluster_size"]
        }
        
        removed = len(clusters) - len(valid_clusters)
        if removed > 0:
            logger.info(f"Removed {removed} clusters for size constraints")
        
        return valid_clusters
    
    def create_cluster_dataset(
        self,
        df: pd.DataFrame,
        clusters: Dict[int, List[int]]
    ) -> List[Dict]:
        """
        Convert clusters to dataset format.
        Returns list of cluster dicts with all necessary data.
        """
        logger.info(f"Creating cluster dataset from {len(clusters)} clusters...")
        
        cluster_data = []
        
        for cluster_id, article_indices in clusters.items():
            articles = df.loc[article_indices].to_dict('records')
            
            # Prepare cluster record
            cluster_record = {
                "cluster_id": cluster_id,
                "num_articles": len(articles),
                "articles": articles,
                "source_indices": article_indices,
                "category": articles[0].get('news_category', 'unknown'),
                "date_range": {
                    "start": str(df.loc[article_indices, 'published_date'].min()),
                    "end": str(df.loc[article_indices, 'published_date'].max()),
                },
            }
            
            # Combine reference summaries (take first available)
            reference_summaries = [
                a.get('human_summary', '')
                for a in articles
                if a.get('human_summary') and len(str(a.get('human_summary', '')).strip()) > 0
            ]
            
            if reference_summaries:
                cluster_record["reference_summary"] = reference_summaries[0]
            else:
                continue  # Skip clusters without reference summaries
            
            cluster_data.append(cluster_record)
        
        logger.info(f"Created {len(cluster_data)} cluster records")
        return cluster_data


class DatasetSplitter:
    """Split clusters into train/val/test sets."""
    
    @staticmethod
    def split_clusters(
        clusters: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split clusters into train/val/test.
        """
        np.random.seed(seed)
        
        indices = np.arange(len(clusters))
        np.random.shuffle(indices)
        
        train_end = int(len(clusters) * train_ratio)
        val_end = train_end + int(len(clusters) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train = [clusters[i] for i in train_indices]
        val = [clusters[i] for i in val_indices]
        test = [clusters[i] for i in test_indices]
        
        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test


def prepare_dataset() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Main pipeline: load, cluster, and split dataset.
    """
    logger.info("="*80)
    logger.info("STARTING DATASET PREPARATION PIPELINE")
    logger.info("="*80)
    
    # Load dataset
    clusterer = NewsSummClusterer(
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=DATASET_CONFIG["similarity_threshold"],
        temporal_window_days=DATASET_CONFIG["temporal_window_days"],
    )
    
    df = clusterer.load_dataset(DATASET_CONFIG["raw_path"])
    
    # Cluster articles
    clusters_dict = clusterer.cluster_articles(df)
    cluster_data = clusterer.create_cluster_dataset(df, clusters_dict)
    
    # Split into train/val/test
    train, val, test = DatasetSplitter.split_clusters(
        cluster_data,
        train_ratio=DATASET_CONFIG["split_ratios"]["train"],
        val_ratio=DATASET_CONFIG["split_ratios"]["val"],
        test_ratio=DATASET_CONFIG["split_ratios"]["test"],
        seed=TRAINING_CONFIG["seed"],
    )
    
    # Save datasets
    save_json(cluster_data, DATASET_CONFIG["processed_path"])
    save_json(train, DATASET_CONFIG["train_clusters_path"])
    save_json(val, DATASET_CONFIG["val_clusters_path"])
    save_json(test, DATASET_CONFIG["test_clusters_path"])
    
    logger.info(f"Saved datasets to {DATASET_CONFIG['processed_path'].parent}")
    logger.info("="*80)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("="*80)
    
    return train, val, test


if __name__ == "__main__":
    set_seed()
    train, val, test = prepare_dataset()
    print(f"\nDataset prepared:")
    print(f"  Train: {len(train)} clusters")
    print(f"  Val: {len(val)} clusters")
    print(f"  Test: {len(test)} clusters")
    print(f"\nSample cluster structure:")
    print(f"  {list(train[0].keys())}")
