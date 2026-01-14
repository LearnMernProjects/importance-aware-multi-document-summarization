"""
Generate Baseline Multi-Document Summaries using facebook/bart-large-cnn
Creates baseline_summaries.csv for evaluation
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# ============================================================
# STEP 1: LOAD MULTI-DOCUMENT CLUSTERS
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOADING MULTI-DOCUMENT CLUSTERS")
print("="*70)

print("Loading news_summ_event_clustered_multidoc.csv...")
df = pd.read_csv('data/processed/news_summ_event_clustered_multidoc.csv')

print(f"  Loaded {len(df)} articles")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# STEP 2: GROUP BY CLUSTER AND PREPARE INPUTS
# ============================================================
print("\n" + "="*70)
print("STEP 2: PREPARING CLUSTER DOCUMENTS FOR SUMMARIZATION")
print("="*70)

# Group by event_cluster_id
clusters = df.groupby('event_cluster_id').agg({
    'headline': list,
    'article_text': list,
    'human_summary': list,
    'news_category': 'first',
    'published_date': list
}).reset_index()

clusters['num_articles_in_cluster'] = clusters['article_text'].apply(len)

print(f"  Total clusters: {len(clusters)}")
print(f"  Articles per cluster: {clusters['num_articles_in_cluster'].min()}-{clusters['num_articles_in_cluster'].max()}")

# Prepare input documents for each cluster
def prepare_cluster_text(articles):
    """Combine multiple articles for multi-document summarization"""
    combined = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(articles)])
    return combined

clusters['cluster_text'] = clusters['article_text'].apply(prepare_cluster_text)

# Prepare reference summaries (use the first human summary or average them)
def prepare_reference_summary(summaries):
    """Use the first available human summary as reference"""
    valid_summaries = [s for s in summaries if pd.notna(s) and len(str(s).strip()) > 0]
    if valid_summaries:
        return valid_summaries[0]
    return ""

clusters['reference_summary'] = clusters['human_summary'].apply(prepare_reference_summary)

# Remove clusters without reference summaries
initial_clusters = len(clusters)
clusters = clusters[clusters['reference_summary'].str.strip() != ""].reset_index(drop=True)
print(f"  Clusters with reference summaries: {len(clusters)}/{initial_clusters}")

# ============================================================
# STEP 3: LOAD SUMMARIZATION MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 3: LOADING SUMMARIZATION MODEL")
print("="*70)

print("Loading facebook/bart-large-cnn...")
print("  (This may take a minute on first load)")

try:
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if False else -1  # Use CPU
    )
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    print("  Attempting to download model...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

# ============================================================
# STEP 4: GENERATE SUMMARIES
# ============================================================
print("\n" + "="*70)
print("STEP 4: GENERATING BASELINE SUMMARIES")
print("="*70)

generated_summaries = []
errors = 0

print(f"Generating summaries for {len(clusters)} clusters...")

for idx, row in tqdm(clusters.iterrows(), total=len(clusters)):
    try:
        cluster_text = row['cluster_text']
        
        # BART has a max length of 1024 tokens, truncate if needed
        # Estimate: ~4 chars per token
        max_chars = 1024 * 4
        if len(cluster_text) > max_chars:
            cluster_text = cluster_text[:max_chars]
        
        # Generate summary
        # min_length: ensure meaningful summary
        # max_length: limit summary length
        summary = summarizer(
            cluster_text,
            max_length=150,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        
        generated_summary = summary[0]['summary_text']
        generated_summaries.append(generated_summary)
        
    except Exception as e:
        print(f"  Error in cluster {idx}: {str(e)[:50]}")
        generated_summaries.append("")
        errors += 1

clusters['baseline_generated_summary'] = generated_summaries

print(f"  ✓ Generated {len(clusters) - errors} summaries")
if errors > 0:
    print(f"  ✗ {errors} errors during generation")

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("STEP 5: SAVING BASELINE SUMMARIES")
print("="*70)

# Select relevant columns
output_df = clusters[[
    'event_cluster_id',
    'baseline_generated_summary',
    'reference_summary',
    'news_category',
    'num_articles_in_cluster'
]].copy()

# Remove rows with empty generated summaries
output_df = output_df[output_df['baseline_generated_summary'].str.strip() != ""]

output_file = 'data/processed/baseline_summaries.csv'
output_df.to_csv(output_file, index=False)

print(f"Baseline summaries saved: {output_file}")
print(f"  Total summaries: {len(output_df)}")
print(f"  Columns: {list(output_df.columns)}")
print(f"  Model: facebook/bart-large-cnn")

# ============================================================
# DIAGNOSTICS
# ============================================================
print("\n" + "="*70)
print("BASELINE GENERATION DIAGNOSTICS")
print("="*70)

print(f"\nGenerated Summary Statistics:")
summary_lengths = output_df['baseline_generated_summary'].str.split().str.len()
ref_lengths = output_df['reference_summary'].str.split().str.len()

print(f"  Generated summaries - words: {summary_lengths.mean():.1f} ± {summary_lengths.std():.1f}")
print(f"  Reference summaries - words: {ref_lengths.mean():.1f} ± {ref_lengths.std():.1f}")
print(f"  Range (generated): {summary_lengths.min():.0f} - {summary_lengths.max():.0f} words")

print(f"\nCategory Distribution:")
for cat in output_df['news_category'].unique():
    count = len(output_df[output_df['news_category'] == cat])
    print(f"  {cat}: {count} clusters")

# ============================================================
# EXECUTION COMPLETE
# ============================================================
print("\n" + "="*70)
print("EXECUTION COMPLETE")
print("="*70)
print(f"\nNext steps:")
print(f"  - Run evaluate_baseline.py to compute ROUGE and BERTScore metrics")
print(f"  - Results will include per-cluster and aggregate performance")
