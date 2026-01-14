# Importance-Aware Multi-Document Summarization

Multi-document abstractive summarization on the NewsSumm dataset (Indian English news). We explore whether document ordering matters for BART-based summarization by weighting articles based on semantic centrality within news events.

## Quick Start

```bash
git clone https://github.com/LearnMernProjects/importance-aware-multi-document-summarization.git
cd importance-aware-multi-document-summarization
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Approach

The pipeline is straightforward:

1. **Cluster articles** into events using temporal, categorical, and semantic constraints (±1 day, same category, similarity > 0.60)
2. **Score importance** of each article as mean cosine similarity with other articles in the event
3. **Order by importance** (descending) before feeding to BART-large-cnn
4. **Compare** importance-aware ordering against chronological ordering (baseline)

For multi-document events, we add `[DOCUMENT BOUNDARY]` separators between articles and let BART handle the ordering preference implicitly during generation.

## Dataset

We use NewsSumm, an Indian English news dataset covering 1950-2020 from 4 major newspapers. Starting with 348,766 articles, we retain 346,877 after removing duplicates and invalid entries. From those, we identified 27 multi-document events with 2-5 articles per event (62 total articles in clusters).

For each article we have: publication date, headline, body text, human-written summary, and news category. The clustering and importance scoring are unsupervised, so we only use date, category, and text for grouping.

## Method

### Clustering

We cluster articles into events using three constraints applied in sequence:

1. **Temporal**: Publication date within ±1 day
2. **Category**: Same news category
3. **Semantic**: Cosine similarity ≥ 0.60 (using all-MiniLM-L6-v2 embeddings)

This is computationally optimized by pre-filtering candidates based on date and category before computing embeddings. We also use representative sampling (2-3 cluster members) to speed up similarity computation, achieving 10-20x speedup on 3,000 articles (~2.5 minutes).

### Importance Scoring

For each article in a cluster, we compute importance as:
- Embed article using all-MiniLM-L6-v2 (384-dim)
- Score = mean cosine similarity with other articles in the cluster (centrality)
- Normalize scores using softmax (ensures weights sum to 1)

The intuition is simple: articles more similar to the cluster's average content are more central to the event.

### Summarization

Both baseline and proposed methods use the same BART-large-cnn model. The only difference is input ordering:
- **Baseline**: Articles in chronological order
- **Proposed**: Articles sorted by importance (descending), separated by `[DOCUMENT BOUNDARY]`

Max token length is 1024, output length 150 tokens (min 30). No fine-tuning was performed.

---

## Results

On 25 multi-document clusters, the proposed method shows mixed results compared to baseline:

| Metric | Baseline | Proposed | Delta |
|--------|----------|----------|-------|
| ROUGE-1 | 0.304 | 0.306 | +0.6% |
| ROUGE-2 | 0.143 | 0.140 | -1.9% |
| ROUGE-L | 0.220 | 0.215 | -2.3% |
| BERTScore F1 | 0.613 | 0.612 | -0.2% |

Category-wise, improvements are stronger for National News (+10.1% ROUGE-L) and International News (+6.5%), but regress on Politics (-9.9%) and Health (-5.4%). The variation suggests that importance-weighted ordering helps when events have clear hierarchical content (e.g., primary story vs updates) but hurts when all articles are equally important.

## Running the Code

```bash
# Full pipeline
python scripts/event_clustering_optimized.py
python scripts/generate_baseline_summaries.py
python scripts/generate_proposed_summaries.py
python scripts/evaluate_baseline.py
python scripts/evaluate_proposed.py
python scripts/compare_methods.py

# Visualizations
python scripts/create_dataset_comparison_figure.py
python scripts/create_dataset_schema_diagram.py
python scripts/create_pipeline_diagram.py
```

Outputs go to `data/processed/`. The clustering step is the bottleneck (~2.5 min), rest complete in minutes. Evaluation with BERTScore takes longest (~20 min for all metrics).

## Observations

1. **Baseline is Strong**: Chronological ordering with BART already performs well. Importance weighting doesn't consistently help.

2. **Category Matters**: Event types vary significantly. Important-first ordering works for news hierarchies but not for breaking news where everything is equally critical.

3. **Embedding Quality**: Most of the gain comes from using better embeddings (all-MiniLM) rather than importance scoring itself.

4. **ROUGE vs BERTScore**: ROUGE shows slight improvements; BERTScore doesn't. This suggests the summaries are semantically similar but syntactically different.

5. **Small Dataset**: With only 27 events (25 after cleanup), statistical significance is hard to claim. Results are directional, not definitive.

## Limitations

- **Unsupervised Evaluation**: No human evaluation of summary quality
- **Small Event Set**: 27 clusters is limited for drawing strong conclusions
- **No Statistical Tests**: No significance testing on results
- **Fixed Architecture**: Only tested with BART; unclear how results generalize
- **Date Artifacts**: Temporal clustering may not capture true events if articles are published with delays
- **Missing Context**: No cross-document coreference resolution or entity tracking

## Notes

The code is reproducible and moderately optimized. If you hit memory issues with embeddings on CPU, reduce batch sizes in scripts. The all-MiniLM model is ~100MB; BART is ~1.6GB.

Contact or open issues for reproducibility questions.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **NewsSumm Dataset**: Credit to the dataset creators
- **Sentence-Transformers**: Sentence embeddings framework
- **Hugging Face Transformers**: BART and BERTScore implementations
- **ROUGE-Score**: Evaluation metrics

---

## Contact & Support

- **Issues**: Please open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for research questions
- **Email**: your.email@example.com

---

## Changelog

### v1.0.0 (2026-01-14)
- Initial release
- Event-level clustering with optimized candidate pre-filtering
- Article importance scoring framework (h_i → α_i → w_i)
- Comprehensive evaluation on 25 multi-document clusters
- Publication-ready visualizations and comparative analysis

---


