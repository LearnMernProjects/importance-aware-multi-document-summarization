"""
Event-Level Clustering for News Articles (Refined)

This script implements an improved event-level clustering pipeline to group news 
articles that report the same real-world event. It is designed to produce meaningful 
multi-document clusters suitable for multi-document summarization experiments.

Key Improvements:
1. Uses sentence-transformers (all-MiniLM-L6-v2) instead of TensorFlow Hub's USE:
   - More efficient and lightweight model
   - Better performance for news article clustering at lower computational cost
   - Easier deployment and integration

2. Lower similarity threshold (0.60 vs 0.75):
   - Stricter threshold produced mostly single-article clusters
   - 0.60 balances precision and recall, forming meaningful multi-document clusters
   - Reflects realistic news event grouping patterns

3. Temporal & Category Constraints (±1 day window):
   - Only articles from the same category within ±1 day can cluster together
   - Reflects real-world news event characteristics
   - Reduces false positives from semantically similar but unrelated events

4. Incremental Clustering:
   - Processes articles in chronological order
   - Assigns each article to an existing cluster or creates a new one
   - Each cluster represents one real-world event

Clustering is based on:
- Semantic similarity (embeddings from sentence-transformers)
- Temporal proximity (±1 day window)
- News category (same category only)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
pd.np.random.seed(42) if hasattr(pd, 'np') else None


def load_and_preprocess_data(csv_path):
    """
    Load the CSV dataset and perform initial preprocessing.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with datetime parsing and null handling.
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    
    # Clean column names (strip whitespace and newlines)
    df.columns = df.columns.str.strip()
    
    # Keep only relevant columns
    relevant_cols = ['article_text', 'headline', 'published_date', 'news_category', 'human_summary']
    df = df[[col for col in relevant_cols if col in df.columns]]
    
    # Parse published_date as datetime with error handling
    df['published_date'] = pd.to_datetime(df['published_date'], format='mixed', 
                                         dayfirst=False, errors='coerce')
    
    # Drop rows with invalid dates
    initial_rows = len(df)
    df = df.dropna(subset=['published_date'])
    invalid_dates = initial_rows - len(df)
    
    if invalid_dates > 0:
        print(f"  Dropped {invalid_dates} rows with invalid dates.")
    
    # Drop rows with missing article_text
    initial_rows = len(df)
    df = df.dropna(subset=['article_text'])
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with missing article_text.")
    
    # Sort by published_date for chronological processing
    df = df.sort_values('published_date').reset_index(drop=True)
    
    print(f"  Loaded {len(df)} articles.")
    return df


def normalize_text(text):
    """
    Normalize text: lowercase and strip whitespace.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Normalized text.
    """
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def create_combined_text(row):
    """
    Combine headline and article_text for better semantic representation.
    
    Args:
        row (pd.Series): A row from the dataframe.
    
    Returns:
        str: Combined and normalized text.
    """
    headline = normalize_text(row['headline'])
    article = normalize_text(row['article_text'])
    combined = f"{headline} {article}"
    return combined


def generate_embeddings(df, model_name="all-MiniLM-L6-v2", batch_size=32):
    """
    Generate embeddings using sentence-transformers.
    
    Why sentence-transformers instead of TensorFlow Hub USE?
    - Lighter weight and more efficient (12M parameters vs 100M+)
    - Faster inference and lower memory footprint
    - Better performance on semantic similarity tasks for news articles
    - Easier integration and deployment
    
    Args:
        df (pd.DataFrame): Input dataframe with articles.
        model_name (str): Name of the sentence-transformers model to use.
        batch_size (int): Batch size for embedding computation.
    
    Returns:
        np.ndarray: Array of embeddings (shape: n_articles x embedding_dim).
    """
    print(f"Loading sentence-transformers model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Create combined text
    print("Creating combined text (headline + article_text)...")
    combined_texts = df.apply(create_combined_text, axis=1).tolist()
    
    # Generate embeddings in batches with progress bar
    print(f"Generating embeddings with batch size {batch_size}...")
    embeddings = model.encode(combined_texts, batch_size=batch_size, 
                             show_progress_bar=True, convert_to_numpy=True)
    
    print(f"  Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")
    return embeddings


def is_within_temporal_window(date1, date2, window_days=1):
    """
    Check if two dates are within a ±window_days window.
    
    Why ±1 day window instead of ±2 days?
    - News events typically break and get reported within a 24-hour cycle
    - Broader windows increase false positives (unrelated events)
    - ±1 day reflects realistic news event grouping patterns
    
    Args:
        date1 (pd.Timestamp): First date.
        date2 (pd.Timestamp): Second date.
        window_days (int): Window size in days (default: 1).
    
    Returns:
        bool: True if dates are within window, False otherwise.
    """
    return abs((date2 - date1).days) <= window_days


def is_same_category(cat1, cat2):
    """
    Check if two articles belong to the same news category.
    
    Args:
        cat1 (str): First category.
        cat2 (str): Second category.
    
    Returns:
        bool: True if categories match, False otherwise.
    """
    return str(cat1).lower().strip() == str(cat2).lower().strip()


def can_cluster_together(idx1, idx2, df, window_days=1):
    """
    Check if two articles can be clustered together based on temporal
    and category constraints.
    
    Why temporal & category filtering?
    - Temporal filtering ensures articles are from the same news cycle
    - Category filtering prevents false positives across unrelated topics
    - Together, they enforce realistic news event grouping constraints
    
    Args:
        idx1 (int): Index of first article.
        idx2 (int): Index of second article.
        df (pd.DataFrame): Dataframe with article metadata.
        window_days (int): Temporal window in days (default: 1).
    
    Returns:
        bool: True if articles can be clustered, False otherwise.
    """
    # Check category
    if not is_same_category(df.loc[idx1, 'news_category'], 
                           df.loc[idx2, 'news_category']):
        return False
    
    # Check temporal window
    if not is_within_temporal_window(df.loc[idx1, 'published_date'],
                                    df.loc[idx2, 'published_date'],
                                    window_days=window_days):
        return False
    
    return True


def incremental_clustering(df, embeddings, similarity_threshold=0.60, 
                          temporal_window=1):
    """
    Perform incremental clustering based on temporal, category, and semantic similarity.
    
    Why a 0.60 similarity threshold instead of 0.75?
    - The initial threshold of 0.75 was too strict, producing mostly single-article clusters
    - 0.60 balances precision and recall:
      - Allows clustering of semantically related articles
      - Still maintains specificity (filters out unrelated articles)
      - Produces meaningful multi-document clusters for summarization tasks
    - Empirically chosen through experimentation with news article data
    
    Incremental Clustering Algorithm:
    1. Iterate through articles in chronological order (ensures determinism)
    2. For each article:
       a. Check all existing clusters for compatibility (category + temporal + similarity)
       b. If similarity >= threshold with any article in cluster: assign to that cluster
       c. Otherwise: create new cluster
    3. Update cluster embeddings as mean of all member embeddings
    
    Args:
        df (pd.DataFrame): Input dataframe with articles.
        embeddings (np.ndarray): Embeddings for all articles.
        similarity_threshold (float): Similarity threshold for clustering (default: 0.60).
        temporal_window (int): Temporal window in days (default: 1).
    
    Returns:
        np.ndarray: Array of cluster IDs (one per article).
    """
    n_articles = len(df)
    cluster_ids = np.full(n_articles, -1, dtype=int)
    next_cluster_id = 0
    
    # Store representative embeddings for each cluster
    cluster_embeddings = {}
    cluster_members = {}  # Track members for diagnostic purposes
    
    print(f"Performing incremental clustering (threshold={similarity_threshold})...")
    
    for i in tqdm(range(n_articles), desc="Clustering articles", unit="article"):
        assigned = False
        best_cluster = -1
        best_similarity = -1
        
        # Try to assign to an existing cluster
        for cluster_id, cluster_emb in cluster_embeddings.items():
            # Check if article can be clustered with this cluster
            # (verify against first member's category and date)
            first_member_idx = cluster_members[cluster_id][0]
            
            if not can_cluster_together(i, first_member_idx, df, 
                                       window_days=temporal_window):
                continue
            
            # Compute cosine similarity with cluster representative embedding
            similarity = cosine_similarity([embeddings[i]], [cluster_emb])[0][0]
            
            if similarity >= similarity_threshold:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
                    assigned = True
        
        if assigned:
            # Assign to existing cluster
            cluster_ids[i] = best_cluster
            # Update cluster embedding as mean of all member embeddings
            cluster_members[best_cluster].append(i)
            member_embeddings = embeddings[cluster_members[best_cluster]]
            cluster_embeddings[best_cluster] = np.mean(member_embeddings, axis=0)
        else:
            # Create new cluster
            cluster_ids[i] = next_cluster_id
            cluster_embeddings[next_cluster_id] = embeddings[i]
            cluster_members[next_cluster_id] = [i]
            next_cluster_id += 1
    
    print(f"Clustering complete. {next_cluster_id} clusters formed.")
    return cluster_ids


def print_diagnostics(df, cluster_ids):
    """
    Print comprehensive clustering diagnostics.
    
    This function validates the clustering by computing:
    - Total number of clusters
    - Average cluster size
    - Cluster size distribution (min, max, mean, median)
    - Number of multi-document clusters (size >= 2)
    - Examples of multi-document clusters for manual inspection
    
    Args:
        df (pd.DataFrame): Input dataframe with articles.
        cluster_ids (np.ndarray): Array of cluster IDs.
    """
    print("\n" + "="*70)
    print("CLUSTER VALIDATION & DIAGNOSTICS")
    print("="*70)
    
    # Total clusters
    n_clusters = len(np.unique(cluster_ids))
    print(f"Total number of clusters: {n_clusters}")
    
    # Average articles per cluster
    avg_articles = len(df) / n_clusters
    print(f"Average number of articles per cluster: {avg_articles:.2f}")
    
    # Distribution statistics
    cluster_counts = pd.Series(cluster_ids).value_counts()
    print(f"\nCluster size distribution:")
    print(f"  Min: {cluster_counts.min()}")
    print(f"  Max: {cluster_counts.max()}")
    print(f"  Mean: {cluster_counts.mean():.2f}")
    print(f"  Median: {cluster_counts.median():.2f}")
    
    # Count multi-document clusters
    multidoc_clusters = (cluster_counts >= 2).sum()
    print(f"\nMulti-document clusters (size >= 2): {multidoc_clusters} ({100*multidoc_clusters/n_clusters:.1f}%)")
    
    # Show example multi-document clusters
    print(f"\nExample multi-document clusters (for manual inspection):")
    print("-"*70)
    
    df_with_clusters = df.copy()
    df_with_clusters['cluster_id'] = cluster_ids
    
    # Get clusters with 2+ articles
    multidoc_cluster_ids = cluster_counts[cluster_counts >= 2].index[:3]  # First 3 examples
    
    for example_idx, cid in enumerate(multidoc_cluster_ids, 1):
        cluster_df = df_with_clusters[df_with_clusters['cluster_id'] == cid]
        print(f"\nCluster {example_idx} (ID: {cid}, Size: {len(cluster_df)}):")
        for row_idx, row in cluster_df.iterrows():
            headline = row['headline'][:70] + "..." if len(str(row['headline'])) > 70 else row['headline']
            print(f"  - [{row['published_date'].strftime('%Y-%m-%d')}] {headline}")
            print(f"    Category: {row['news_category']}")
    
    # Show sample single-article cluster
    singleart_cluster_ids = cluster_counts[cluster_counts == 1].index[0]
    single_cluster_df = df_with_clusters[df_with_clusters['cluster_id'] == singleart_cluster_ids]
    print(f"\nSample single-article cluster (ID: {singleart_cluster_ids}):")
    row = single_cluster_df.iloc[0]
    headline = row['headline'][:70] + "..." if len(str(row['headline'])) > 70 else row['headline']
    print(f"  - [{row['published_date'].strftime('%Y-%m-%d')}] {headline}")
    print(f"    Category: {row['news_category']}")


def main():
    """
    Main execution function for the event clustering pipeline.
    
    Workflow:
    1. Load and preprocess the CSV dataset
    2. Generate embeddings using sentence-transformers
    3. Perform incremental clustering with temporal/category constraints
    4. Validate clustering results and print diagnostics
    5. Save full clustered dataset
    6. Filter and save multi-document clusters (size >= 2) for summarization
    """
    # Configuration
    CSV_PATH = "data/processed/newssumm_clean.csv"
    OUTPUT_PATH_FULL = "data/processed/news_summ_event_clustered_refined.csv"
    OUTPUT_PATH_MULTIDOC = "data/processed/news_summ_event_clustered_multidoc.csv"
    
    # Clustering parameters
    SIMILARITY_THRESHOLD = 0.60  # Reduced from 0.75 to capture meaningful clusters
    TEMPORAL_WINDOW = 1  # days - realistic news event window
    BATCH_SIZE = 32
    MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight, efficient sentence-transformers model
    
    try:
        # Step 1: Load and preprocess data
        print("\n" + "="*70)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*70)
        df = load_and_preprocess_data(CSV_PATH)
        print(f"Successfully loaded and preprocessed {len(df)} articles.")
        
        # Step 2: Generate embeddings
        print("\n" + "="*70)
        print("STEP 2: GENERATING EMBEDDINGS")
        print("="*70)
        embeddings = generate_embeddings(df, model_name=MODEL_NAME, batch_size=BATCH_SIZE)
        
        # Step 3: Perform incremental clustering
        print("\n" + "="*70)
        print("STEP 3: INCREMENTAL CLUSTERING WITH CONSTRAINTS")
        print("="*70)
        cluster_ids = incremental_clustering(df, embeddings,
                                            similarity_threshold=SIMILARITY_THRESHOLD,
                                            temporal_window=TEMPORAL_WINDOW)
        
        # Step 4: Add cluster IDs to dataframe
        df['event_cluster_id'] = cluster_ids
        
        # Step 5: Print comprehensive diagnostics
        print_diagnostics(df, cluster_ids)
        
        # Step 6: Save full clustered dataset
        print("\n" + "="*70)
        print("STEP 4: SAVING RESULTS")
        print("="*70)
        df.to_csv(OUTPUT_PATH_FULL, index=False)
        print(f"✓ Full clustered dataset saved to: {OUTPUT_PATH_FULL}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Step 7: Filter for multi-document clusters (size >= 2) for summarization
        print("\nFiltering for multi-document clusters (size >= 2)...")
        cluster_sizes = df['event_cluster_id'].value_counts()
        multidoc_cluster_ids = cluster_sizes[cluster_sizes >= 2].index
        df_multidoc = df[df['event_cluster_id'].isin(multidoc_cluster_ids)].copy()
        
        df_multidoc.to_csv(OUTPUT_PATH_MULTIDOC, index=False)
        print(f"✓ Multi-document dataset saved to: {OUTPUT_PATH_MULTIDOC}")
        print(f"  Total articles in multi-document clusters: {len(df_multidoc)}")
        print(f"  Total clusters: {len(multidoc_cluster_ids)}")
        print(f"  Average articles per cluster: {len(df_multidoc) / len(multidoc_cluster_ids):.2f}")
        
        # Final summary
        print("\n" + "="*70)
        print("EXECUTION COMPLETE")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Input file: {CSV_PATH}")
        print(f"  Total articles processed: {len(df)}")
        print(f"  Total clusters formed: {len(np.unique(cluster_ids))}")
        print(f"  Multi-document clusters: {len(multidoc_cluster_ids)}")
        print(f"  Articles in multi-document clusters: {len(df_multidoc)}")
        print(f"\nOutput files:")
        print(f"  1. Full dataset: {OUTPUT_PATH_FULL}")
        print(f"  2. Multi-document (for summarization): {OUTPUT_PATH_MULTIDOC}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
