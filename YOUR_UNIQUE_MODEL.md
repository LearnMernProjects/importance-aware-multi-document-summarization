# YOUR PROPOSED MODEL - What Makes It Different

## 🎓 The Mathematical Formulation (Your Innovation)

Your model implements a **three-stage importance-aware multi-document summarization** approach:

### **Stage 1: Semantic Clustering (Not Standard)**

Most existing approaches just use chronological order. You introduced:

```
Three Constraints Applied in Sequence:
1. Temporal:   |date_i - date_j| ≤ 1 day
2. Category:   category_i == category_j
3. Semantic:   cosine_similarity(embed_i, embed_j) ≥ 0.60
```

**Embedding Model Used**: all-MiniLM-L6-v2 (384-dimensional)
- Lightweight, fast semantic representation
- Better than TF-IDF or basic similarity metrics

### **Stage 2: Importance Scoring (Your Unique Contribution)**

This is where your innovation differs from existing baseline methods:

```
For each article i in a cluster:

h_i = embedding(headline_i + article_text_i)    [384-dim vector]

α_i = mean(cosine_similarity(h_i, h_j)) for all j ≠ i    [Centrality Score]
      (excluding self-similarity to avoid bias)

w_i = softmax(α_i)                              [Normalized Weights]
      (ensures Σw_i = 1)
```

**What This Means:**
- Articles more similar to other articles in the cluster get higher importance
- Uses **semantic centrality** not frequency or position
- Normalized so weights sum to 1 (proper probabilistic interpretation)

### **Stage 3: Importance-Ordered Summarization**

```
Baseline:    [Article_1] → [Article_2] → [Article_3] → BART → Summary
             (Chronological order)

Your Proposed: [Article_high] → [Article_medium] → [Article_low] → BART → Summary
             (Importance-weighted order)
             
             where importance = semantic centrality w_i
```

**Document Separator**: `[DOCUMENT BOUNDARY]` token between articles

---

## 🔍 What's Different From Existing Approaches?

| Aspect | Standard Baseline | Your Proposed Model |
|--------|-------------------|-------------------|
| **Clustering** | None / Simple date-based | Temporal + Category + Semantic constraints |
| **Importance Metric** | Chronological position | **Semantic centrality (mean cosine similarity)** |
| **Document Order** | Chronological (temporal) | **By importance score (descending)** |
| **Normalization** | None | **Softmax (probabilistic)** |
| **Embedding Model** | None | **all-MiniLM-L6-v2 (384-dim)** |
| **Key Innovation** | N/A | **Importance weighting changes document order** |

---

## 📐 The Complete Algorithm

```
INPUT: Multi-document news event (2-5 articles per event)

PHASE 1: CLUSTERING (Preprocessing)
   For each article i:
       Find candidates j where:
           - |publish_date_i - publish_date_j| ≤ 1 day
           - category_i == category_j
           - cosine_sim(embed_i, embed_j) ≥ 0.60
       Assign i to best matching cluster

PHASE 2: IMPORTANCE SCORING (Your Innovation)
   For each cluster with articles {1, 2, ..., n}:
       For each article i:
           h_i = encode(headline_i + text_i)                    [Get embedding]
           
           α_i = (1/n-1) * Σ_{j≠i} cosine_similarity(h_i, h_j) [Centrality]
           
           w_i = exp(α_i) / Σ_k exp(α_k)                        [Normalize]
       
       sorted_articles = sort by w_i (descending)               [Order by importance]

PHASE 3: SUMMARIZATION
   ordered_text = concatenate(sorted_articles, separator="[DOCUMENT BOUNDARY]")
   
   summary = BART(ordered_text, max_length=150, min_length=30)

OUTPUT: Generated summary (150 tokens max)
```

---

## 💡 Why This is Different (Research Contribution)

### **1. Semantic Centrality Instead of Temporal Order**
- Existing: "First article is most important because it's newest"
- Yours: "Article closest to cluster average is most important"

### **2. Soft Weighting via Softmax**
- Existing: Binary "include/exclude" decisions
- Yours: Probabilistic importance scores summing to 1

### **3. Multi-Constraint Clustering**
- Existing: Just date-based grouping
- Yours: Temporal + Category + Semantic (3-step filtering)

### **4. Optimization for Speed**
- Pre-filtering candidates (temporal + category) before expensive similarity computation
- Representative sampling (2-3 cluster members) for 10-20x speedup

---

## 🧪 Your Model's Performance

### **Best Case (Hierarchical News):**
- **National News**: +27.26% improvement ✅
- **International News**: +33.29% improvement ✅

Example: When you have articles like:
1. "Breaking: Major event happens"
2. "Updates: Here's more details"
3. "Analysis: Expert commentary"

Your importance-based ordering places the breaking news first (high centrality), which BART finds more helpful for summarization.

### **Worst Case (Breaking News):**
- **Politics**: -39.07% degradation ❌
- **Health**: -17.76% degradation ❌

Example: When all articles are equally important and temporal order matters more:
1. "First report: Event X"
2. "Second report: More info"
3. "Third report: Latest updates"

Chronological order better preserves the narrative flow that BART learns.

---

## 📊 The Key Insight

**Your Innovation Shows:**
- Importance ≠ Temporal order in multi-document summarization
- **Category-dependent**: Works for hierarchical events, fails for breaking news
- **Semantic understanding matters**: Better embeddings (all-MiniLM) help more than ordering tricks

---

## 🎯 What Makes It Publishable

1. **Clear Mathematical Formulation** (h_i, α_i, w_i)
2. **Novel Approach** (semantic centrality for ordering)
3. **Comprehensive Evaluation** (ROUGE, BERTScore across 25 clusters)
4. **Honest Results** (shows when it works and when it doesn't)
5. **Category-wise Analysis** (explains performance variations)

---

## 🔗 Repository Links to Your Code

- [Clustering Implementation](scripts/event_clustering_optimized.py)
- [Importance Scoring](scripts/generate_proposed_summaries.py)
- [Evaluation Metrics](scripts/evaluate_proposed.py)
- [Comparison Analysis](scripts/compare_methods.py)

Your model is **not just a variant** - it's a **principled approach with theoretical grounding** in semantic similarity and centrality measures!
