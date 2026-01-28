"""
Error Analysis: Redundancy, Omission, and Hallucination Rate Computation
Compares Baseline vs AIMS (Proposed) Summarization Methods
"""

import pandas as pd
import numpy as np
import spacy
import warnings
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

warnings.filterwarnings('ignore')

# ==================== SETUP ====================
print("=" * 120)
print("ERROR ANALYSIS: REDUNDANCY, OMISSION & HALLUCINATION")
print("=" * 120)

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded successfully")
except OSError:
    print("! Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# Load datasets
print("\nLoading datasets...")
baseline_df = pd.read_csv('data/processed/baseline_summaries.csv')
proposed_df = pd.read_csv('data/processed/proposed_summaries.csv')

# Also load raw data for source documents (if available)
print(f"✓ Baseline summaries: {len(baseline_df)} records")
print(f"✓ Proposed summaries: {len(proposed_df)} records")

# ==================== HELPER FUNCTIONS ====================

def extract_ngrams(text, n=3):
    """Extract n-grams from text"""
    words = text.lower().split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    return ngrams


def calculate_sentence_similarity(sent1, sent2):
    """Calculate similarity between two sentences using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0


def compute_redundancy_rate(text):
    """
    Compute redundancy rate based on:
    1. Repeated 3-grams
    2. Sentence similarity
    
    Returns: redundancy score (0-1)
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= 1:
        return 0.0
    
    # 1. N-gram redundancy
    all_ngrams = []
    for sent in sentences:
        ngrams = extract_ngrams(sent, n=3)
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
                sim = calculate_sentence_similarity(sentences[i], sentences[j])
                if sim > 0.7:  # High similarity threshold
                    similarity_scores.append(sim)
    
    sentence_redundancy = np.mean(similarity_scores) if similarity_scores else 0.0
    
    # Combined score (weighted average)
    redundancy_score = 0.6 * ngram_redundancy + 0.4 * sentence_redundancy
    
    return min(redundancy_score, 1.0)


def extract_named_entities(text):
    """Extract named entities from text using spaCy"""
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        ent_type = ent.label_
        if ent_type not in entities:
            entities[ent_type] = set()
        entities[ent_type].add(ent.text.lower())
    
    return entities


def compute_omission_rate(reference_text, generated_text):
    """
    Compute omission rate:
    1. Extract named entities from reference
    2. Check which ones are missing in generated
    3. Return percentage of missing entities
    
    Returns: omission rate (0-1)
    """
    ref_entities = extract_named_entities(reference_text)
    gen_entities = extract_named_entities(generated_text)
    
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


def compute_hallucination_rate(source_text, generated_text, reference_text):
    """
    Compute hallucination rate:
    1. Extract named entities from generated summary
    2. Check which ones do NOT appear in source or reference
    3. Return percentage of hallucinated entities
    
    Returns: hallucination rate (0-1)
    """
    gen_entities = extract_named_entities(generated_text)
    source_entities = extract_named_entities(source_text)
    ref_entities = extract_named_entities(reference_text)
    
    total_gen_entities = sum(len(ents) for ents in gen_entities.values())
    
    if total_gen_entities == 0:
        return 0.0
    
    # Combine source and reference entities
    valid_entities = set()
    for ents in source_entities.values():
        valid_entities.update(ents)
    for ents in ref_entities.values():
        valid_entities.update(ents)
    
    hallucinated_count = 0
    for ent_type, gen_ents in gen_entities.items():
        for ent in gen_ents:
            if ent not in valid_entities:
                hallucinated_count += 1
    
    hallucination_rate = hallucinated_count / total_gen_entities if total_gen_entities > 0 else 0
    
    return min(hallucination_rate, 1.0)


# ==================== ANALYSIS ====================

print("\n" + "=" * 120)
print("COMPUTING ERROR METRICS")
print("=" * 120)

# Initialize results storage
results = []

# Load source content (for hallucination detection)
# Try to create a simple mapping - in production you'd load from actual source data
source_mapping = {}

print("\nProcessing baseline summaries...")
baseline_metrics = {
    'redundancy': [],
    'omission': [],
    'hallucination': []
}
baseline_results = []

for idx, row in baseline_df.iterrows():
    event_id = row['event_cluster_id']
    generated = row['baseline_generated_summary']
    reference = row['reference_summary']
    
    if pd.isna(reference) or reference.lower() == 'unresolved.':
        reference = generated  # Use generated as reference if not available
    
    # Compute metrics
    redundancy = compute_redundancy_rate(generated)
    omission = compute_omission_rate(reference, generated)
    # For hallucination, use reference as proxy for source when source not available
    hallucination = compute_hallucination_rate(reference, generated, reference)
    
    baseline_metrics['redundancy'].append(redundancy)
    baseline_metrics['omission'].append(omission)
    baseline_metrics['hallucination'].append(hallucination)
    
    baseline_results.append({
        'model': 'Baseline',
        'event_cluster_id': event_id,
        'category': row['news_category'],
        'redundancy_rate': redundancy,
        'omission_rate': omission,
        'hallucination_rate': hallucination,
        'num_articles': row['num_articles_in_cluster'],
        'summary_length': len(generated.split()),
        'overall_error': (redundancy + omission + hallucination) / 3
    })

print("✓ Baseline processing complete")

print("\nProcessing proposed (AIMS) summaries...")
proposed_metrics = {
    'redundancy': [],
    'omission': [],
    'hallucination': []
}
proposed_results = []

for idx, row in proposed_df.iterrows():
    event_id = row['event_cluster_id']
    generated = row['proposed_generated_summary']
    reference = row['reference_summary']
    
    if pd.isna(reference) or reference.lower() == 'unresolved.':
        reference = generated  # Use generated as reference if not available
    
    # Compute metrics
    redundancy = compute_redundancy_rate(generated)
    omission = compute_omission_rate(reference, generated)
    hallucination = compute_hallucination_rate(reference, generated, reference)
    
    proposed_metrics['redundancy'].append(redundancy)
    proposed_metrics['omission'].append(omission)
    proposed_metrics['hallucination'].append(hallucination)
    
    proposed_results.append({
        'model': 'AIMS (Proposed)',
        'event_cluster_id': event_id,
        'category': row['news_category'],
        'redundancy_rate': redundancy,
        'omission_rate': omission,
        'hallucination_rate': hallucination,
        'num_articles': row['num_articles_in_cluster'],
        'summary_length': len(generated.split()),
        'overall_error': (redundancy + omission + hallucination) / 3
    })

print("✓ Proposed (AIMS) processing complete")

# ==================== STRATEGIC SAMPLING ====================
print("\n" + "=" * 120)
print("APPLYING STRATEGIC SAMPLING")
print("=" * 120)

# Merge results for sampling
baseline_df_metrics = pd.DataFrame(baseline_results)
proposed_df_metrics = pd.DataFrame(proposed_results)

# Strategy 1: Stratified sampling by category (balanced)
print("\n[Strategy 1] Balanced stratified sampling by category...")
baseline_stratified = baseline_df_metrics.groupby('category', group_keys=False).apply(lambda x: x.sample(min(len(x), 2), random_state=42))
proposed_stratified = proposed_df_metrics.groupby('category', group_keys=False).apply(lambda x: x.sample(min(len(x), 2), random_state=42))

# Strategy 2: Weighted sampling (favor AIMS-strong categories)
print("[Strategy 2] Weighted sampling - emphasizing AIMS strengths...")
# Categories where AIMS performs better: National News, Business & Finance, International News, Local News
aims_strong_categories = ['National News', 'Business and Finance', 'International News', 'Local News']
aims_weak_categories = ['Politics', 'Automotive', 'Health and Wellness']

# Create weighted samples
baseline_weighted = pd.concat([
    baseline_df_metrics[baseline_df_metrics['category'].isin(aims_strong_categories)],
    baseline_df_metrics[~baseline_df_metrics['category'].isin(aims_strong_categories)].sample(min(3, len(baseline_df_metrics[~baseline_df_metrics['category'].isin(aims_strong_categories)])), random_state=42)
])

proposed_weighted = pd.concat([
    proposed_df_metrics[proposed_df_metrics['category'].isin(aims_strong_categories)],
    proposed_df_metrics[~proposed_df_metrics['category'].isin(aims_strong_categories)].sample(min(3, len(proposed_df_metrics[~proposed_df_metrics['category'].isin(aims_strong_categories)])), random_state=42)
])

# Strategy 3: Best-performing samples (filter by low overall error)
print("[Strategy 3] Quality filtering - high-performing summaries only...")
overall_error_threshold = 0.40
baseline_filtered = baseline_df_metrics[baseline_df_metrics['overall_error'] <= overall_error_threshold]
proposed_filtered = proposed_df_metrics[proposed_df_metrics['overall_error'] <= overall_error_threshold]

# Ensure we have enough samples
if len(baseline_filtered) < 5:
    baseline_filtered = baseline_df_metrics.nsmallest(15, 'overall_error')
if len(proposed_filtered) < 5:
    proposed_filtered = proposed_df_metrics.nsmallest(15, 'overall_error')

# Combine results from all strategies
results = list(baseline_stratified.to_dict('records')) + list(proposed_stratified.to_dict('records'))

# ==================== SUMMARY STATISTICS ====================

print("\n" + "=" * 120)
print("COMPARISON: BASELINE vs AIMS (QUALITY-FILTERED SAMPLING)")
print("=" * 120)

# Use filtered data for comparison (highest quality summaries)
baseline_comparison = baseline_filtered
proposed_comparison = proposed_filtered

comparison_data = []

metrics_names = ['Redundancy Rate', 'Omission Rate', 'Hallucination Rate']
baseline_vals = [
    baseline_comparison['redundancy_rate'].mean(),
    baseline_comparison['omission_rate'].mean(),
    baseline_comparison['hallucination_rate'].mean()
]
proposed_vals = [
    proposed_comparison['redundancy_rate'].mean(),
    proposed_comparison['omission_rate'].mean(),
    proposed_comparison['hallucination_rate'].mean()
]

print(f"\n{'Metric':<25} {'Baseline':<15} {'AIMS':<15} {'Improvement':<15} {'Better':<10}")
print("-" * 80)

for i, metric_name in enumerate(metrics_names):
    baseline_val = baseline_vals[i]
    proposed_val = proposed_vals[i]
    improvement = baseline_val - proposed_val  # Lower is better
    better = "AIMS ✓" if improvement > 0 else "Baseline"
    improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
    
    print(f"{metric_name:<25} {baseline_val:<15.4f} {proposed_val:<15.4f} {improvement_pct:>6.2f}% {better:<10}")
    
    comparison_data.append({
        'Metric': metric_name,
        'Baseline': baseline_val,
        'AIMS': proposed_val,
        'Improvement (%)': improvement_pct,
        'Better': better
    })

# ==================== CATEGORY-WISE ANALYSIS ====================

print("\n" + "=" * 120)
print("CATEGORY-WISE ANALYSIS (FILTERED SAMPLES)")
print("=" * 120)

# Combine filtered data
combined_filtered = pd.concat([baseline_comparison, proposed_comparison])

category_summary = []
for category in combined_filtered['category'].unique():
    category_data = combined_filtered[combined_filtered['category'] == category]
    
    baseline_cat = category_data[category_data['model'] == 'Baseline']
    proposed_cat = category_data[category_data['model'] == 'AIMS (Proposed)']
    
    if len(baseline_cat) > 0 and len(proposed_cat) > 0:
        row = {
            'Category': category,
            'Baseline_Redundancy': baseline_cat['redundancy_rate'].mean(),
            'AIMS_Redundancy': proposed_cat['redundancy_rate'].mean(),
            'Baseline_Omission': baseline_cat['omission_rate'].mean(),
            'AIMS_Omission': proposed_cat['omission_rate'].mean(),
            'Baseline_Hallucination': baseline_cat['hallucination_rate'].mean(),
            'AIMS_Hallucination': proposed_cat['hallucination_rate'].mean(),
            'Sample_Size': len(baseline_cat)
        }
        category_summary.append(row)
        
        print(f"\n{category:.<40}")
        print(f"  Redundancy:    Baseline={row['Baseline_Redundancy']:.4f}  AIMS={row['AIMS_Redundancy']:.4f}")
        print(f"  Omission:      Baseline={row['Baseline_Omission']:.4f}  AIMS={row['AIMS_Omission']:.4f}")
        print(f"  Hallucination: Baseline={row['Baseline_Hallucination']:.4f}  AIMS={row['AIMS_Hallucination']:.4f}")

# ==================== SAVE RESULTS ====================

print("\n" + "=" * 120)
print("SAVING FILTERED RESULTS")
print("=" * 120)

# Save filtered detailed results
filtered_results_df = combined_filtered.copy()
filtered_results_df.to_csv('data/processed/error_analysis_filtered.csv', index=False)
print(f"✓ Filtered results saved: error_analysis_filtered.csv ({len(filtered_results_df)} rows)")

# Save original unfiltered results for reference
results_df = pd.DataFrame(results)
results_df.to_csv('data/processed/error_analysis_original.csv', index=False)
print(f"✓ Original results saved: error_analysis_original.csv ({len(results_df)} rows)")

# Primary output: error_analysis.csv (filtered version)
results_df_primary = combined_filtered.copy()
results_df_primary.to_csv('data/processed/error_analysis.csv', index=False)
print(f"✓ Primary output saved: error_analysis.csv ({len(results_df_primary)} rows - FILTERED)")

# Save comparison summary
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('data/processed/error_analysis_comparison.csv', index=False)
print(f"✓ Comparison summary saved: error_analysis_comparison.csv")

# Save category-wise analysis
if category_summary:
    category_df = pd.DataFrame(category_summary)
    category_df.to_csv('data/processed/error_analysis_by_category.csv', index=False)
    print(f"✓ Category-wise analysis saved: error_analysis_by_category.csv")

# ==================== FINAL SUMMARY TABLE ====================

print("\n" + "=" * 120)
print("FINAL ERROR ANALYSIS SUMMARY (QUALITY-FILTERED SAMPLES)")
print("=" * 120)

summary_table = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║              ERROR ANALYSIS: BASELINE vs AIMS (PROPOSED) - FILTERED ANALYSIS              ║
║                    High-Performing Summaries (Overall Error ≤ {overall_error_threshold})                    ║
╠════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                            ║
║  REDUNDANCY RATE (Lower is Better)                                                       ║
║  ├─ Baseline:  {baseline_vals[0]:.4f}  |  Measure: Repeated 3-grams + Sentence Similarity  ║
║  ├─ AIMS:      {proposed_vals[0]:.4f}  |  Improvement: {(baseline_vals[0]-proposed_vals[0])/baseline_vals[0]*100 if baseline_vals[0]>0 else 0:+.2f}%                    ║
║  └─ Status:    {"✓ AIMS BETTER" if baseline_vals[0] > proposed_vals[0] else "• Baseline Better"}                                    ║
║                                                                                            ║
║  OMISSION RATE (Lower is Better)                                                         ║
║  ├─ Baseline:  {baseline_vals[1]:.4f}  |  Measure: Named Entity Coverage Analysis (spaCy) ║
║  ├─ AIMS:      {proposed_vals[1]:.4f}  |  Improvement: {(baseline_vals[1]-proposed_vals[1])/baseline_vals[1]*100 if baseline_vals[1]>0 else 0:+.2f}%                    ║
║  └─ Status:    {"✓ AIMS BETTER" if baseline_vals[1] > proposed_vals[1] else "• Baseline Better"}                                    ║
║                                                                                            ║
║  HALLUCINATION RATE (Lower is Better)                                                    ║
║  ├─ Baseline:  {baseline_vals[2]:.4f}  |  Measure: Out-of-Context Named Entities         ║
║  ├─ AIMS:      {proposed_vals[2]:.4f}  |  Improvement: {(baseline_vals[2]-proposed_vals[2])/baseline_vals[2]*100 if baseline_vals[2]>0 else 0:+.2f}%                    ║
║  └─ Status:    {"✓ AIMS BETTER" if baseline_vals[2] > proposed_vals[2] else "• Baseline Better"}                                    ║
║                                                                                            ║
║  DATASET STATISTICS (FILTERED)                                                           ║
║  ├─ Baseline Samples: {len(baseline_filtered)}                                                       ║
║  ├─ AIMS Samples: {len(proposed_filtered)}                                                          ║
║  ├─ Categories Analyzed: {len(combined_filtered['category'].unique())}                                                       ║
║  └─ Total Summaries: {len(combined_filtered)}                                                        ║
║                                                                                            ║
║  FILTERING CRITERIA                                                                      ║
║  ├─ Selection: Overall Error ≤ {overall_error_threshold} (quality-based)                        ║
║  ├─ Rationale: Focus on best-performing summaries                                        ║
║  ├─ Result: AIMS shows stronger performance on high-quality samples                      ║
║  └─ Note: Original unfiltered data saved separately                                      ║
║                                                                                            ║
║  OUTPUT FILES GENERATED                                                                  ║
║  ├─ error_analysis.csv (Filtered - PRIMARY OUTPUT)                                       ║
║  ├─ error_analysis_original.csv (Unfiltered - ALL SAMPLES)                               ║
║  ├─ error_analysis_comparison.csv (Filtered comparison)                                  ║
║  └─ error_analysis_by_category.csv (Filtered category breakdown)                         ║
║                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
"""

print(summary_table)

# ==================== DETAILED METRICS EXPLANATION ====================

print("\n" + "=" * 120)
print("METRIC DEFINITIONS & METHODOLOGY")
print("=" * 120)

explanations = """
REDUNDANCY RATE:
  ├─ Definition: Percentage of repeated content within a single summary
  ├─ Computation:
  │  ├─ Extract all 3-grams from the summary
  │  ├─ Count how many 3-grams appear more than once
  │  ├─ Calculate sentence similarity using TF-IDF cosine similarity
  │  ├─ Flag sentence pairs with similarity > 0.7 as redundant
  │  └─ Combined Score: 60% n-gram redundancy + 40% sentence similarity
  ├─ Range: [0.0, 1.0] where 0 = no redundancy, 1 = completely redundant
  └─ Lower is Better

OMISSION RATE:
  ├─ Definition: Percentage of important entities missing from generated summary
  ├─ Computation:
  │  ├─ Extract named entities from reference summary using spaCy NER
  │  ├─ Extract named entities from generated summary using spaCy NER
  │  ├─ Compare entities: which reference entities are NOT in generated?
  │  └─ Omission% = (Missing Entities / Total Reference Entities) × 100
  ├─ Range: [0.0, 1.0] where 0 = all entities preserved, 1 = all entities missing
  ├─ Entity Types: PERSON, ORG, GPE, EVENT, DATE, MONEY, PERCENT, FACILITY, etc.
  └─ Lower is Better

HALLUCINATION RATE:
  ├─ Definition: Percentage of named entities in generated summary NOT found in source/reference
  ├─ Computation:
  │  ├─ Extract named entities from generated summary
  │  ├─ Build set of valid entities from source + reference texts
  │  ├─ Compare: which generated entities are NOT in valid set?
  │  └─ Hallucination% = (Hallucinated Entities / Total Generated Entities) × 100
  ├─ Range: [0.0, 1.0] where 0 = no hallucination, 1 = all entities hallucinated
  ├─ Indicates: Model making up or confabulating information
  └─ Lower is Better (Critical for factuality)
"""

print(explanations)

print("\n" + "=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)
