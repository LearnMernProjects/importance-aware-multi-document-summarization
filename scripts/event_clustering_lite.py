"""
Event-Level Clustering for News Articles (Optimized for Large Datasets)

This script implements event-level clustering to group news articles that report
the same real-world event. Clustering is based on:
- Semantic similarity (embeddings)
- Temporal proximity (±2 day window)
- News category
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import sys

warnings.filterwarnings('ignore')

# Suppress TensorFlow info messages
tf.get_logger().setLevel('ERROR')


def load_and_preprocess_data(csv_path, sample_size=5000):
    """
    Load the CSV dataset with memory-efficient sampling.
    
    Args:
        csv_path (str): Path to the CSV file.
        sample_size (int): Number of samples to load.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    print("Loading dataset...")
    
    try:
        # Read CSV with only needed columns
        df = pd.read_csv(
            csv_path, 
            usecols=['article_text', 'headline', 'published_date', 'news_category', 'human_summary'],
            dtype={'article_text': str, 'headline': str, 'news_category': str},
            on_bad_lines='skip',
            engine='python'
        )
    except Exception:
        # Fallback: read all columns
        df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
        cols_needed = ['article_text', 'headline', 'published_date', 'news_category', 'human_summary']
        df = df[[col for col in cols_needed if col in df.columns]]
    
    print(f"  Loaded {len(df)} articles initially.")
    
    # Parse published_date
    df['published_date'] = pd.to_datetime(df['published_date'], format='mixed', 
                                         dayfirst=False, errors='coerce')
    
    # Drop invalid dates
    initial = len(df)
    df = df.dropna(subset=['published_date'])
    if len(df) < initial:
        print(f"  Dropped {initial - len(df)} rows with invalid dates.")
    
    # Drop missing article_text
    initial = len(df)
    df = df.dropna(subset=['article_text'])
    if len(df) < initial:
        print(f"  Dropped {initial - len(df)} rows with missing article_text.")
    
    # Sample if too large
    if len(df) > sample_size:
        print(f"  Sampling {sample_size} articles from {len(df)} for efficient clustering...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Sort chronologically
    df = df.sort_values('published_date').reset_index(drop=True)
    
    print(f"  Final dataset: {len(df)} articles.")
    return df


def normalize_text(text):
    """Normalize text: lowercase and strip whitespace."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def create_combined_text(row):
    """Combine headline and article_text."""
    headline = normalize_text(row['headline'])
    article = normalize_text(row['article_text'])
    return f"{headline} {article}"


def generate_embeddings(df, batch_size=16):
    """
    Generate embeddings using TensorFlow Hub Universal Sentence Encoder.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        batch_size (int): Batch size for processing.
    
    Returns:
        np.ndarray: Embeddings array.
    """
    print("Loading TensorFlow Hub Universal Sentence Encoder...")
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    
    try:
        model = hub.load(module_url)
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Using lightweight encoder instead...")
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
        model = hub.load(module_url)
    
    # Create combined text
    print("Preparing text for embeddings...")
    combined_texts = df.apply(create_combined_text, axis=1).tolist()
    
    # Generate embeddings
    print(f"Generating embeddings ({len(combined_texts)} texts, batch size {batch_size})...")
    embeddings_list = []
    
    for i in range(0, len(combined_texts), batch_size):
        if i % max(100, batch_size * 4) == 0:
            progress = (i / len(combined_texts)) * 100
            print(f"  Progress: {progress:.1f}%")
        
        batch_texts = combined_texts[i:i + batch_size]
        batch_embeddings = model(batch_texts)
        embeddings_list.append(batch_embeddings.numpy())
    
    embeddings = np.vstack(embeddings_list)
    print(f"  Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")
    
    return embeddings


def can_cluster_together(idx1, idx2, df, window_days=2):
    """Check if two articles can be clustered."""
    # Check category
    cat1 = str(df.loc[idx1, 'news_category']).lower().strip()
    cat2 = str(df.loc[idx2, 'news_category']).lower().strip()
    if cat1 != cat2:
        return False
    
    # Check temporal window
    date1 = df.loc[idx1, 'published_date']
    date2 = df.loc[idx2, 'published_date']
    if abs((date2 - date1).days) > window_days:
        return False
    
    return True


def incremental_clustering(df, embeddings, similarity_threshold=0.75, temporal_window=2):
    """
    Perform incremental clustering.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        embeddings (np.ndarray): Embeddings array.
        similarity_threshold (float): Similarity threshold.
        temporal_window (int): Temporal window in days.
    
    Returns:
        np.ndarray: Cluster IDs.
    """
    n_articles = len(df)
    cluster_ids = np.full(n_articles, -1, dtype=int)
    next_cluster_id = 0
    
    cluster_embeddings = {}
    cluster_members = {}
    
    print(f"Performing incremental clustering (threshold={similarity_threshold})...")
    
    for i in range(n_articles):
        if i % max(100, n_articles // 10) == 0:
            progress = (i / n_articles) * 100
            print(f"  Progress: {progress:.1f}% ({i}/{n_articles})")
        
        assigned = False
        best_cluster = -1
        best_similarity = -1
        
        # Try existing clusters
        for cluster_id, cluster_emb in cluster_embeddings.items():
            first_member_idx = cluster_members[cluster_id][0]
            
            if not can_cluster_together(i, first_member_idx, df, window_days=temporal_window):
                continue
            
            # Compute similarity
            similarity = cosine_similarity([embeddings[i]], [cluster_emb])[0][0]
            
            if similarity >= similarity_threshold:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
                    assigned = True
        
        if assigned:
            cluster_ids[i] = best_cluster
            cluster_members[best_cluster].append(i)
            # Update cluster embedding
            member_embeddings = embeddings[cluster_members[best_cluster]]
            cluster_embeddings[best_cluster] = np.mean(member_embeddings, axis=0)
        else:
            # Create new cluster
            cluster_ids[i] = next_cluster_id
            cluster_embeddings[next_cluster_id] = embeddings[i]
            cluster_members[next_cluster_id] = [i]
            next_cluster_id += 1
    
    print(f"  Clustering complete. {next_cluster_id} clusters formed.")
    return cluster_ids


def print_diagnostics(df, cluster_ids):
    """Print clustering diagnostics."""
    print("\n" + "="*60)
    print("CLUSTERING DIAGNOSTICS")
    print("="*60)
    
    n_clusters = len(np.unique(cluster_ids))
    print(f"Total number of clusters: {n_clusters}")
    print(f"Average articles per cluster: {len(df) / n_clusters:.2f}")
    
    cluster_counts = pd.Series(cluster_ids).value_counts()
    print(f"\nCluster size distribution:")
    print(f"  Min: {cluster_counts.min()}, Max: {cluster_counts.max()}, Median: {cluster_counts.median():.2f}")
    
    print(f"\nSample results (5 rows):")
    print("-"*60)
    sample_df = df[['headline', 'news_category', 'published_date']].head(5).copy()
    sample_df['event_cluster_id'] = cluster_ids[:5]
    
    for idx, row in sample_df.iterrows():
        print(f"\nArticle {idx}: Cluster {row['event_cluster_id']}")
        print(f"  Category: {row['news_category']}")
        print(f"  Date: {row['published_date']}")
        headline = row['headline'][:80] + "..." if len(str(row['headline'])) > 80 else row['headline']
        print(f"  Headline: {headline}")


def main():
    """Main execution."""
    CSV_PATH = "data/processed/newssumm_clean.csv"
    OUTPUT_PATH = "data/processed/news_summ_event_clustered.csv"
    SAMPLE_SIZE = 5000
    BATCH_SIZE = 16
    SIMILARITY_THRESHOLD = 0.75
    TEMPORAL_WINDOW = 2
    
    try:
        print("\n" + "="*60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*60)
        df = load_and_preprocess_data(CSV_PATH, sample_size=SAMPLE_SIZE)
        
        print("\n" + "="*60)
        print("STEP 2: GENERATING EMBEDDINGS")
        print("="*60)
        embeddings = generate_embeddings(df, batch_size=BATCH_SIZE)
        
        print("\n" + "="*60)
        print("STEP 3: EVENT-LEVEL CLUSTERING")
        print("="*60)
        cluster_ids = incremental_clustering(df, embeddings, 
                                            similarity_threshold=SIMILARITY_THRESHOLD,
                                            temporal_window=TEMPORAL_WINDOW)
        
        df['event_cluster_id'] = cluster_ids
        
        print_diagnostics(df, cluster_ids)
        
        print("\n" + "="*60)
        print("STEP 4: SAVING RESULTS")
        print("="*60)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Clustered dataset saved to: {OUTPUT_PATH}")
        
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
