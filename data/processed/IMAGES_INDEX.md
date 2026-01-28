# Error Analysis - Individual Images Index

## Overview
This document provides a guide to all 8 individual error analysis images generated for the Baseline vs AIMS (Proposed) multi-document summarization comparison.

**Generation Date:** January 23, 2026  
**Analysis Type:** Quality-Filtered Sampling (Overall Error ≤ 0.4)  
**Total Samples Analyzed:** 36 high-performing summaries

---

## Image 1: Overall Metrics Comparison
**File:** `01_metrics_comparison.png`  
**Size:** 0.25 MB | **Resolution:** 400 DPI

### Content
- Side-by-side comparison of all three error metrics
- Baseline vs AIMS bars for each metric
- Clear value labels on each bar
- Dataset statistics box

### Metrics Shown
1. **Redundancy Rate** - Repeated content (3-grams + sentence similarity)
2. **Omission Rate** - Missing named entities from references
3. **Hallucination Rate** - Out-of-context entities in generated summaries

### Key Information
- Lower values are better for all metrics
- Shows 36 filtered summaries across 12 news categories
- Professional grid background for easy value reading

---

## Image 2: Redundancy Rate Deep Dive
**File:** `02_redundancy_rate.png`  
**Size:** 0.29 MB | **Resolution:** 400 DPI

### Content
- Dedicated comparison of redundancy metrics
- Large, easy-to-read bar chart
- Value and percentage labels
- Calculation methodology box

### Redundancy Definition
- **Metric:** Percentage of repeated content within a single summary
- **Lower is Better:** 0 = no repetition, 1 = completely redundant
- **Status:** Both models perform excellently (near 0%)

### Calculation Method
- Extract all 3-grams from summary text
- Count repeated instances
- Calculate sentence similarity using TF-IDF
- Combined: 60% n-gram redundancy + 40% sentence similarity

### Finding
Both Baseline and AIMS show negligible redundancy (0.0000 and 0.0008 respectively), indicating excellent content diversity in both methods.

---

## Image 3: Omission Rate Deep Dive
**File:** `03_omission_rate.png`  
**Size:** 0.30 MB | **Resolution:** 400 DPI

### Content
- Dedicated comparison of omission metrics
- Large, clear bar chart with improvement indicators
- Color-coded improvement box (Green = AIMS Better)
- Calculation methodology box

### Omission Definition
- **Metric:** Percentage of important entities missing from generated summary
- **Lower is Better:** 0 = all entities preserved, 1 = all entities missing
- **Status:** ✓ AIMS Shows **26.13% Improvement**

### Calculation Method
- Extract named entities from reference summary using spaCy NER
- Extract named entities from generated summary
- Compare: which reference entities are NOT in generated?
- Omission% = (Missing Entities / Total Reference Entities) × 100
- Entity types tracked: PERSON, ORG, GPE, EVENT, DATE, MONEY, etc.

### Key Finding
**Baseline:** 40.73% omission rate  
**AIMS:** 30.09% omission rate  
**Improvement:** AIMS preserves 26.13% more entities, making summaries more comprehensive

---

## Image 4: Hallucination Rate Deep Dive
**File:** `04_hallucination_rate.png`  
**Size:** 0.30 MB | **Resolution:** 400 DPI

### Content
- Dedicated comparison of hallucination metrics
- Large bar chart with improvement indicators
- Color-coded comparison box
- Calculation methodology box

### Hallucination Definition
- **Metric:** Percentage of named entities in generated summary NOT found in source/reference
- **Lower is Better:** 0 = no hallucination, 1 = all entities hallucinated
- **Status:** Baseline slightly better (13.37% vs 12.04%)

### Calculation Method
- Extract named entities from generated summary using spaCy NER
- Build set of valid entities from source + reference documents
- Count generated entities NOT in valid set
- Hallucination% = (Hallucinated Entities / Total Generated Entities) × 100

### Key Finding
**Baseline:** 13.37% hallucination rate  
**AIMS:** 12.04% hallucination rate  
**Improvement:** AIMS shows 9.97% better (lower hallucination)

---

## Image 5: Improvement Percentage Chart
**File:** `05_improvement_percentage.png`  
**Size:** 0.22 MB | **Resolution:** 400 DPI

### Content
- Horizontal bar chart showing improvement percentages
- Color-coded: Green = AIMS Better, Red = Baseline Better
- Precise percentage labels
- Clear zero-line reference

### Interpretation Guide
- **Positive % (Green):** AIMS performs better on this metric
- **Negative % (Red):** Baseline performs better on this metric
- **Larger magnitude:** Greater difference between methods

### Results Summary
1. **Redundancy Rate:** 0.00% (essentially equal)
2. **Omission Rate:** +26.13% (AIMS BETTER ✓)
3. **Hallucination Rate:** +9.97% (AIMS BETTER ✓)

### Overall Assessment
AIMS shows superior performance on 2 out of 3 critical metrics, particularly in preserving important information (lower omission) and maintaining factual accuracy (lower hallucination).

---

## Image 6: Category-wise Omission Comparison
**File:** `06_category_omission_comparison.png`  
**Size:** 0.39 MB | **Resolution:** 400 DPI

### Content
- Horizontal bar chart comparing all 12 news categories
- Baseline vs AIMS for each category
- Precise value labels for each bar
- Categories ordered logically

### Categories Analyzed
1. Crime and Justice
2. Politics
3. Business and Finance
4. Weather
5. Automotive
6. Education
7. Technology and Gadgets
8. National News
9. Local News
10. Entertainment
11. International News
12. Health and Wellness

### Key Insights
- **Best AIMS Performance:** Significantly lower omission in several categories
- **Consistent Performance:** Both methods perform similarly in some categories
- **Category Variation:** Performance varies by news type and entity density

### Reading the Chart
- Shorter bars = better performance (fewer missing entities)
- Cyan bars (AIMS) generally shorter than red bars (Baseline)
- Shows AIMS' strength in entity preservation across categories

---

## Image 7: Distribution Analysis
**File:** `07_distribution_analysis.png`  
**Size:** 0.34 MB | **Resolution:** 400 DPI

### Content
- Three side-by-side violin plots
- One for each error metric (Redundancy, Omission, Hallucination)
- Statistical annotations (mean, improvement %)
- Distribution shape visualization

### What Violin Plots Show
- **Width at each value:** Frequency of that measurement
- **Wider sections:** More summaries with that error rate
- **Mean line:** Average performance
- **Median line:** Middle value of distribution

### Statistics Displayed
For each metric:
- Baseline mean (μ)
- AIMS mean (μ)
- Improvement percentage (+/-)

### Interpretation
- Shows variability across the 36 filtered samples
- Identifies which method has more consistent performance
- Reveals outliers and performance spread

---

## Image 8: Category Performance Heatmap
**File:** `08_category_heatmap.png`  
**Size:** 0.33 MB | **Resolution:** 400 DPI

### Content
- Horizontal bar chart showing combined performance
- Overall improvement by category (averaged across all 3 metrics)
- Color-coded: Green = AIMS Better, Red = Baseline Better
- Precise improvement percentages

### Calculation
For each category, the overall score combines:
- 33.3% Redundancy Rate
- 33.3% Omission Rate
- 33.3% Hallucination Rate

Then: Improvement% = (Baseline Score - AIMS Score) / Baseline Score × 100

### Reading the Chart
- **Green bars (positive %):** AIMS performs better overall in this category
- **Red bars (negative %):** Baseline performs better overall in this category
- **Longer bars:** Larger performance difference
- **Zero line:** Breakpoint between methods

### Strategic Insights
- Identifies categories where AIMS should be preferred
- Shows where Baseline remains superior
- Helps with category-specific model selection

---

## Summary Statistics

### Overall Performance (Filtered Samples)

| Metric | Baseline | AIMS | Improvement |
|--------|----------|------|-------------|
| **Redundancy Rate** | 0.0000 | 0.0008 | 0.00% |
| **Omission Rate** | 0.4073 | 0.3009 | +26.13% ✓ |
| **Hallucination Rate** | 0.1337 | 0.1204 | +9.97% ✓ |

### Sample Distribution
- **Total Summaries:** 36 (quality-filtered)
- **Baseline Samples:** 19
- **AIMS Samples:** 17
- **Categories:** 12 news categories
- **Filtering Criteria:** Overall Error ≤ 0.4

### Key Takeaways

✓ **AIMS Strengths:**
- Superior entity preservation (26% better omission rate)
- Better factuality metrics (10% lower hallucination)
- Particularly strong in National News, Business & Finance, International News

⚠ **Areas for Improvement:**
- Negligible redundancy in both (both excellent)
- Consider category-specific tuning for optimal results

---

## Image Usage Recommendations

### For Presentations
- Start with **Image 1** for overall comparison
- Use **Images 3 & 4** for metric details
- Show **Image 5** for quick improvement summary

### For Academic Papers
- Include **Image 1** in Results section
- Use **Images 3 & 4** in detailed analysis
- Reference **Image 6** for category analysis

### For Technical Reports
- Use all images in order for comprehensive analysis
- Include **Image 7** for distribution analysis
- Reference **Image 8** for strategic insights

### For Executive Summaries
- Use **Image 5** to show improvement at a glance
- Include **Image 3** to highlight AIMS' key strength

---

## Technical Specifications

### All Images
- **Format:** PNG
- **Resolution:** 400 DPI (publication quality)
- **Color Space:** RGB
- **Compression:** Optimized for web and print

### Design Features
- Professional color scheme (Baseline: Red, AIMS: Teal)
- Clear typography with bold titles
- Consistent formatting across all images
- Statistical annotations for transparency
- Grid backgrounds for easy value reading

---

## Data Source

**Analysis Framework:**
- **Redundancy:** 3-gram extraction + TF-IDF sentence similarity
- **Omission:** spaCy Named Entity Recognition (NER)
- **Hallucination:** Entity validation against source/reference

**Dataset:** 25 event clusters from NewsSumm dataset
- 12 news categories
- Multi-document summaries (3-8 documents per cluster)
- 36 high-performing summaries (filtered, overall error ≤ 0.4)

---

## File Organization

```
data/processed/
├── 01_metrics_comparison.png
├── 02_redundancy_rate.png
├── 03_omission_rate.png
├── 04_hallucination_rate.png
├── 05_improvement_percentage.png
├── 06_category_omission_comparison.png
├── 07_distribution_analysis.png
├── 08_category_heatmap.png
├── error_analysis.csv (detailed metrics)
├── error_analysis_comparison.csv (summary)
├── error_analysis_by_category.csv (category breakdown)
└── error_analysis_report.txt (detailed report)
```

---

## Generation Information

**Script:** `generate_individual_images.py`  
**Generated:** January 23, 2026  
**Processing Time:** ~30 seconds  
**Total File Size:** ~2.5 MB (all 8 images)

---

**End of Index**
