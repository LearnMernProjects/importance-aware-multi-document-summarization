# 🎯 QUICK REFERENCE - MODELS & TECHNIQUES USED

## ⚠️ IMPORTANT: ROUGE is NOT a Model, It's an EVALUATION METRIC

---

## 📊 COMPLETE LIST OF ALL MODELS & TECHNIQUES

### **1. SUMMARIZATION MODEL** 🤖

**Model: facebook/bart-large-cnn**
```
Purpose:   Generate summaries from articles
Type:      Sequence-to-Sequence Transformer
Input:     1024 tokens (multi-document articles)
Output:    150 tokens (summary)
Why:       - Best for news summarization
           - Pre-trained on news data
           - Handles multi-document input
```

---

### **2. EMBEDDING MODEL** 🔢

**Model: all-MiniLM-L6-v2**
```
Purpose:   Convert articles to semantic vectors
Type:      Sentence Transformer
Output:    384-dimensional embeddings
Why:       - Fast processing (22MB model)
           - Good semantic quality
           - Optimized for similarity
```

---

### **3. EVALUATION METRICS** 📈

#### **ROUGE-1 (Unigram Overlap)**
```
Measures:  Single word matching
Formula:   Overlapping words / Total reference words
Range:     0 to 1 (higher is better)
Why:       Captures basic information coverage
```

#### **ROUGE-2 (Bigram Overlap)**
```
Measures:  Two-word phrase matching
Formula:   Overlapping bigrams / Total reference bigrams
Range:     0 to 1 (higher is better)
Why:       Captures phrase-level consistency
```

#### **ROUGE-L (Longest Common Subsequence)**
```
Measures:  Longest consecutive matching sequence
Formula:   LCS / Reference length
Range:     0 to 1 (higher is better)
Why:       Captures longer semantic units
```

#### **BERTScore F1**
```
Measures:  Semantic similarity (contextual)
Type:      Uses BERT embeddings
Range:     0 to 1 (higher is better)
Why:       - Better than ROUGE
           - Handles synonyms
           - Contextual understanding
```

---

### **4. MATHEMATICAL TECHNIQUES** 🧮

#### **Cosine Similarity**
```
Formula:   cos(θ) = (A · B) / (||A|| ||B||)
Purpose:   Measure angle between embedding vectors
Range:     -1 to +1 (0.6+ considered similar)
Why:       - Fast computation
           - Works well with embeddings
           - Normalized scale
```

#### **Softmax Normalization**
```
Formula:   w_i = exp(α_i) / Σ exp(α_j)
Purpose:   Convert scores to probabilities (sum = 1)
Why:       - Probabilistic interpretation
           - Emphasizes importance differences
           - Industry standard
```

---

### **5. PREPROCESSING TECHNIQUES** 🛠️

#### **Multi-Constraint Clustering**
```
Constraint 1: Temporal (±1 day)
Constraint 2: Category (same category only)
Constraint 3: Semantic (cosine similarity ≥ 0.60)
Why:       Find related articles for multi-doc events
```

#### **Document Ordering** (Your Innovation)
```
Baseline:  Chronological order (by date)
Proposed:  Importance order (by semantic centrality)
Why:       Test if importance helps summarization
```

---

## 📋 COMPLETE SUMMARY TABLE

```
┌─────────────────────┬──────────────┬─────────────────────────────┐
│   Component         │     Type     │      Why Used               │
├─────────────────────┼──────────────┼─────────────────────────────┤
│ BART-large-cnn      │ Model        │ Best news summarization     │
│ all-MiniLM-L6-v2    │ Model        │ Fast semantic embeddings    │
│ ROUGE-1             │ Metric       │ Word overlap measurement    │
│ ROUGE-2             │ Metric       │ Phrase matching             │
│ ROUGE-L             │ Metric       │ Sequence matching           │
│ BERTScore F1        │ Metric       │ Semantic similarity         │
│ Cosine Similarity   │ Technique    │ Fast vector comparison      │
│ Softmax             │ Technique    │ Probability normalization   │
│ Multi-Constraint    │ Technique    │ Article clustering          │
│ Importance Order    │ Innovation   │ YOUR UNIQUE CONTRIBUTION    │
└─────────────────────┴──────────────┴─────────────────────────────┘
```

---

## 🎯 WHY THESE SPECIFIC CHOICES?

### **ROUGE is NOT a Model - It's an Evaluation Metric**

**What ROUGE Does:**
- Compares generated summary to reference summary
- Counts overlapping words/phrases
- Produces scores (0-1 scale)
- Helps evaluate quality automatically

**Why Use ROUGE?**
```
✅ Industry standard (used in 99% of papers)
✅ Automatic (no manual annotation needed)
✅ Reproducible (same results every time)
✅ Interpretable (easy to understand)
✅ Three variants (captures different aspects)
```

**What ROUGE Can't Do:**
```
❌ Doesn't understand synonyms
❌ Doesn't capture semantic meaning
❌ Just word overlap, not understanding
❌ That's why we also use BERTScore
```

---

## 📊 FLOW: FROM DATA TO RESULTS

```
Raw Articles (346,877)
        ↓
[All-MiniLM embeddings]
        ↓
[Cosine Similarity clustering]
        ↓
Multi-document Events (27)
        ↓
[Softmax importance scoring] ← YOUR INNOVATION
        ↓
Two Document Orders:
  - Baseline: Chronological
  - Proposed: By Importance
        ↓
[BART summarization (both)]
        ↓
Generated Summaries (25 clusters × 2 methods = 50)
        ↓
[ROUGE-1, ROUGE-2, ROUGE-L evaluation]
[BERTScore evaluation]
        ↓
Results & Insights
```

---

## ✅ RESEARCH METHODOLOGY CHECKLIST

- ✅ **Summarization Model**: State-of-the-art (BART)
- ✅ **Embedding Model**: Efficient & effective (all-MiniLM)
- ✅ **Primary Metrics**: Standard practice (ROUGE)
- ✅ **Secondary Metrics**: Semantic evaluation (BERTScore)
- ✅ **Mathematical Techniques**: Well-established (Cosine, Softmax)
- ✅ **Innovation**: Your importance-weighted ordering
- ✅ **Evaluation**: Rigorous & reproducible
- ✅ **Publication Quality**: Industry-standard approach

---

## 🎓 CONCLUSION

**Your project uses:**

1. **BART Model** - For summarization
2. **MiniLM Model** - For embeddings
3. **ROUGE Metrics** - For evaluation (✅ It's a metric, not a model!)
4. **BERTScore** - For semantic evaluation
5. **Cosine Similarity** - For clustering
6. **Softmax** - For importance weighting
7. **Your Innovation** - Importance-ordered summarization

**Why this combination?**
- ✅ Proven models
- ✅ Rigorous evaluation
- ✅ Reproducible methodology
- ✅ Publication-ready approach
- ✅ Your unique contribution stands out

**Ready for publication! ✅**
