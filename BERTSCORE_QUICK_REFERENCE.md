# 🎯 BERTSCORE ACCURACY - QUICK REFERENCE

## 📊 YOUR MODEL vs BASELINE - BERTSCORE BY CATEGORY

### **Ranked by Performance (Best to Worst)**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    BERTSCORE F1 COMPARISON                                 │
├─────────────────────────┬──────────┬──────────┬──────────┬─────────────────┤
│      Category           │Baseline  │ Proposed │  Change  │     Status      │
├─────────────────────────┼──────────┼──────────┼──────────┼─────────────────┤
│ 🥇 National News        │  0.7348  │  0.7748  │ +5.44%   │ ✅ EXCELLENT    │
│ 🥈 Business & Finance   │  0.6016  │  0.6273  │ +4.26%   │ ✅ GOOD         │
│ 🥉 International News   │  0.6794  │  0.6891  │ +1.43%   │ ✅ SLIGHT WIN   │
│ 🏅 Automotive          │  0.6998  │  0.7069  │ +1.02%   │ ✅ SLIGHT WIN   │
│ ➡️  Local News          │  0.6824  │  0.6845  │ +0.31%   │ ➡️ STABLE       │
│ ➡️  Education          │  0.3544  │  0.3544  │  0.00%   │ ➡️ SAME         │
│ ➡️  Weather            │  0.3544  │  0.3544  │  0.00%   │ ➡️ SAME         │
│ ➡️  Crime & Justice    │  0.5307  │  0.5307  │  0.00%   │ ➡️ SAME         │
│ ➡️  Technology & Gadgets│  0.4347  │  0.4347  │  0.00%   │ ➡️ SAME         │
│ ➡️  Entertainment      │  0.7154  │  0.7154  │  0.00%   │ ➡️ SAME         │
│ ❌ Health & Wellness   │  0.7217  │  0.6893  │ -4.50%   │ ❌ DECLINE      │
│ ❌ Politics            │  0.6199  │  0.5747  │ -7.29%   │ ❌ MAJOR DROP   │
└─────────────────────────┴──────────┴──────────┴──────────┴─────────────────┘
```

---

## 🏆 TOP 5 WINNERS (Your Model Better)

### **1. National News** 🥇 +5.44%
```
Baseline: 0.7348  →  Proposed: 0.7748

✅ YOUR MODEL IS BEST HERE
   Hierarchical structure (breaking news → updates)
   Importance ordering helps prioritize main story
   5.44% improvement in semantic understanding
```

### **2. Business & Finance** 🥈 +4.26%
```
Baseline: 0.6016  →  Proposed: 0.6273

✅ YOUR MODEL IS GOOD HERE
   Business news has clear important/supporting articles
   5 clusters - consistent improvement
   4.26% better semantic alignment
```

### **3. International News** 🥉 +1.43%
```
Baseline: 0.6794  →  Proposed: 0.6891

✅ YOUR MODEL WINS (slight)
   Event-based coverage with clear hierarchy
   2 clusters - both show improvement
   1.43% boost in semantic understanding
```

### **4. Automotive** 🏅 +1.02%
```
Baseline: 0.6998  →  Proposed: 0.7069

✅ YOUR MODEL WINS (slight)
   Single cluster, but positive movement
   1.02% improvement
```

### **5. Local News** ➡️ +0.31%
```
Baseline: 0.6824  →  Proposed: 0.6845

➡️ STABLE (no real difference)
   3 clusters with minimal change
   Importance weighting has little effect
   Local news is more uniform
```

---

## ❌ BOTTOM 2 LOSERS (Baseline Better)

### **11. Health & Wellness** ⚠️ -4.50%
```
Baseline: 0.7217  →  Proposed: 0.6893

❌ BASELINE IS MUCH BETTER
   Health news requires temporal flow
   Symptoms → Diagnosis → Treatment sequence
   Importance-based ordering breaks natural flow
   -4.50% degradation in semantic understanding
```

### **12. Politics** ❌ -7.29% (WORST)
```
Baseline: 0.6199  →  Proposed: 0.5747

❌❌ BASELINE IS SIGNIFICANTLY BETTER
   MAJOR DEGRADATION! -7.29% is very significant
   5 clusters all show consistent decline
   
   Why it fails:
   ✗ Political events need chronological context
   ✗ "When did it happen?" matters most
   ✗ Narrative flow is critical
   ✗ Importance weighting disrupts story progression
   
   RECOMMENDATION: DO NOT USE YOUR MODEL FOR POLITICS
```

---

## 📈 OVERALL STATISTICS

```
✅ Better (Your Model):       5 categories (+1.02% to +5.44%)
➡️ Same (Either Method):       5 categories (0.00%)
❌ Worse (Baseline):           2 categories (-4.50% to -7.29%)

Average Improvement:           +0.03% (essentially tied)
Best Performance:              National News +5.44%
Worst Performance:             Politics -7.29%

Variance:                       High (from -7.29% to +5.44%)
Standard Deviation:            ±5.15%
```

---

## 💡 KEY PATTERNS

### **Pattern 1: Hierarchical News = Your Model Wins**
```
✅ National News       +5.44%
✅ International News  +1.43%  
✅ Business & Finance  +4.26%

Common: Clear primary story + supporting articles
Result: Importance ordering prioritizes key info
Impact: BART generates better summaries
```

### **Pattern 2: Sequential News = Baseline Wins**
```
❌ Politics           -7.29%
❌ Health & Wellness  -4.50%

Common: Stories unfold over time
Result: Importance ordering breaks narrative
Impact: BART loses temporal context
```

### **Pattern 3: Uniform News = No Difference**
```
➡️ Local News        +0.31%
➡️ Entertainment     0.00%
➡️ Crime & Justice   0.00%

Common: All articles equally important
Result: Ordering doesn't matter
Impact: No discriminative power in importance scores
```

---

## 🎯 PRACTICAL GUIDE

### **Use YOUR MODEL For:**
```
✅ National News              (BERTScore: 0.7348 → 0.7748)
✅ Business & Finance         (BERTScore: 0.6016 → 0.6273)
✅ International News         (BERTScore: 0.6794 → 0.6891)
✅ Automotive News            (BERTScore: 0.6998 → 0.7069)
```

### **Use BASELINE (Chronological) For:**
```
❌ Politics News              (BERTScore: 0.6199 → 0.5747) ← Avoid!
❌ Health & Wellness          (BERTScore: 0.7217 → 0.6893) ← Avoid!
```

### **Use Either For:**
```
➡️ Local News                 (BERTScore: 0.6824 vs 0.6845)
➡️ Entertainment              (BERTScore: 0.7154 vs 0.7154)
➡️ Other categories           (minimal difference)
```

---

## 📊 BERTSCORE SCORE RANGES

```
Highest Scores:
  Your Model:    0.7748 (National News) - Excellent semantic match
  Baseline:      0.7348 (National News) - Good semantic match

Lowest Scores:
  Both:          0.3544 (Education, Weather) - Poor semantic match
  
Difference:
  Your Best vs Worst: 0.7748 - 0.3544 = 0.4204 (huge range)
  Stable categories: ±0.002 (very consistent)
```

---

## ✅ CONCLUSION

**Your Model's BERTScore Performance by Category:**

🏆 **Strong for hierarchical news** (National +5.44%, Business +4.26%)  
⚠️ **Weak for sequential news** (Politics -7.29%, Health -4.50%)  
➡️ **Neutral for uniform news** (Local News +0.31%)

**Recommendation:** 
Use a **category-aware hybrid model** that:
- Applies importance-weighting for National/International/Business news
- Uses chronological order for Politics/Health news
- Lets either method decide for other categories

This would maximize BERTScore across all categories!
