# 🚀 Google Colab GPU Training Guide

## ✅ QUICK START (5 Minutes Setup)

### Step 1: Prepare Your Files
1. **Upload your project to Google Drive:**
   - Upload the entire "Suvidha" folder to your Google Drive root
   - Required files:
     - `data/processed/newssumm_clean.csv` (articles)
     - `data/processed/news_summ_event_clustered.csv` (clusters)

### Step 2: Open the Colab Notebook
1. Go to: **https://colab.research.google.com**
2. Click **"Upload"** tab
3. Upload this file: `Train_All_11_Models_Google_Colab.ipynb`
4. Open it

### Step 3: Configure GPU
1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** (Free, ~2-4 hours training)
   - Or **A100 GPU** (Faster if available, ~30-60 mins)
3. Click **Save**

### Step 4: Run All Cells
1. Click **Runtime** → **Run all** (Or press Ctrl+F9)
2. First time will download models (~3-5 GB total)
3. Training starts automatically

---

## ⏱️ EXPECTED TIMING

| GPU Type | Time | Cost |
|----------|------|------|
| **T4 GPU** | 2-4 hours | FREE ✅ |
| **A100 GPU** | 30-60 mins | FREE (limited quota) |
| **V100 GPU** | 1-2 hours | $25-50 |
| **A100 (paid)** | 20-30 mins | $1-2/hr |

---

## 📊 WHAT YOU GET

✅ **CSV Results File:**
- All 11 models × 11 metrics
- Ready for Excel/analysis

✅ **5 Publication-Quality Images:**
1. ROUGE comparison (all 3 metrics)
2. BERTScore + Faithfulness comparison
3. Error metrics (Redundancy, Omission, Hallucination)
4. Complete metrics heatmap
5. AIMS improvement analysis

✅ **Saved to Google Drive:**
- `MyDrive/Suvidha/data/processed/11_models_training_results/`

---

## 📋 11 MODELS TRAINED

### Baseline Models (5)
✅ PEGASUS - `google/pegasus-arxiv`
✅ LED - `allenai/led-base-16384` (Longformer)
✅ BigBird - `google/bigbird-pegasus-large-arxiv`
✅ PRIMERA - `allenai/primera`
✅ LongT5 - `google/long-t5-tglobal-base`

### Advanced Approaches (5)
✅ GraphSum - Graph-based summarization
✅ Instruction-LLM - Instruction-tuned large model
✅ Factuality-Aware - Generator + Verifier
✅ Event-Aware - Event detection + weighting
✅ Benchmark-LLM - Standard LLM baseline

### Your Innovation (1)
⭐ **AIMS** - Article-level Importance-aware Multi-document Summarization

---

## 🔧 TROUBLESHOOTING

### Issue: "No GPU available"
**Solution:** 
- Click Runtime → Change runtime type
- Make sure "GPU" is selected
- Restart kernel and try again

### Issue: "Out of Memory" error
**Solution:**
- The notebook reduces batch sizes automatically
- If it still fails, reduce the number of clusters in Step 4
- Change `[:30]` to `[:15]` to use fewer test clusters

### Issue: Model download hangs
**Solution:**
- This is normal - models are large (500MB - 2GB each)
- Give it 10-15 minutes
- Don't interrupt the cell

### Issue: Results not in Drive
**Solution:**
- Results are saved to: `MyDrive/Suvidha/data/processed/11_models_training_results/`
- Refresh your Google Drive (F5)
- The folder takes 1-2 minutes to appear

---

## 💾 SAVE RESULTS

After training completes:

### Option 1: Direct Download
1. Expand folder icon (left side)
2. Navigate to results folder
3. Right-click CSV → Download

### Option 2: From Google Drive
1. Open Google Drive
2. Go to: `MyDrive/Suvidha/data/processed/11_models_training_results/`
3. Download entire folder with all images

### Option 3: View in Colab
- Results are displayed directly in the notebook
- Take screenshots of charts if needed

---

## 📊 RESULTS PREVIEW

**Example Output Format:**

```
Model          | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Faithfulness
PEGASUS        |  0.4500 |  0.2800 |  0.4200 |   0.8800  |   0.9200
LED            |  0.4700 |  0.3000 |  0.4400 |   0.8900  |   0.9300
BigBird        |  0.4300 |  0.2600 |  0.4000 |   0.8600  |   0.9000
PRIMERA        |  0.4600 |  0.2900 |  0.4300 |   0.8800  |   0.9200
LongT5         |  0.4500 |  0.2700 |  0.4100 |   0.8700  |   0.9100
GraphSum       |  0.4400 |  0.2800 |  0.4200 |   0.8750  |   0.9150
Instruction    |  0.4200 |  0.2500 |  0.4000 |   0.8600  |   0.9000
Factuality     |  0.4550 |  0.2850 |  0.4250 |   0.8850  |   0.9250
Event-Aware    |  0.4650 |  0.2950 |  0.4350 |   0.8900  |   0.9300
Benchmark-LLM  |  0.4100 |  0.2400 |  0.3900 |   0.8500  |   0.8900
AIMS ⭐        |  0.5200 |  0.3500 |  0.5000 |   0.9200  |   0.9500
                           ↑                      ↑              ↑
                         BEST              BEST (Semantic)    BEST
```

---

## 🎯 NEXT STEPS AFTER TRAINING

1. **Analyze Results:**
   - Open CSV in Excel
   - Sort by ROUGE-1, ROUGE-2, etc.
   - Find which model(s) perform best

2. **Use Images in Paper:**
   - All PNG files are publication-ready (300 DPI)
   - Include in results section
   - Reference in methods section

3. **Statistical Analysis:**
   - Calculate standard deviations
   - Run t-tests on results
   - Determine statistical significance

4. **Write Discussion:**
   - Compare AIMS vs top 3 baselines
   - Explain why AIMS performs better
   - Discuss trade-offs (accuracy vs speed)

---

## ❓ FAQ

**Q: Will my data be stored on Google servers?**
A: Only while Colab is running. Once you disconnect, everything is deleted except what you save to Drive.

**Q: Can I pause and resume?**
A: Yes, but the notebook session closes after 30 mins of inactivity. Better to run it all at once.

**Q: How much storage do I need?**
A: ~5-10 GB in Drive (models cache + results)

**Q: Can I modify the notebook?**
A: Yes! It's fully customizable. Change model names, metrics, or evaluation logic.

**Q: What if I want to train only specific models?**
A: Edit the `MODELS_CONFIG` dictionary in Step 5. Remove models you don't want.

---

## 📚 SUPPORT

**Issues?** Check Colab cell output for specific error messages. Most are self-explanatory.

**Questions?** Review comments in the notebook cells - they explain each step.

**Want to customize?** You can:
- Add more evaluation metrics
- Change model configurations
- Adjust training parameters
- Modify visualization styles

---

## ✨ You're Ready!

**Next action:**
1. ✅ Upload Suvidha folder to Google Drive
2. ✅ Upload this notebook to Colab
3. ✅ Change Runtime to GPU
4. ✅ Click "Run all"
5. ✅ Wait 2-4 hours
6. ✅ Download results & images

**The notebook handles everything else!**

---

Generated: January 27, 2026
For: Importance-Aware Multi-Document Summarization Project
