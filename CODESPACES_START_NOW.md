# ⚡ GITHUB CODESPACES - START NOW! (5 Minutes)

## 🚀 QUICKEST PATH TO TRAINING

### **STEP 1: Fork the Repository (2 minutes)**

```
1. Go to: https://github.com/LearnMernProjects/importance-aware-multi-document-summarization

2. Click: Fork (top right corner)
   
3. Choose account: Select your GitHub account
   
4. Click: Create fork
   
5. WAIT for fork to complete (~30 seconds)
   You'll see: "LearnMernProjects/importance-aware-multi-document-summarization"
   becomes "YOUR_USERNAME/importance-aware-multi-document-summarization"
```

✅ **Fork Complete!**

---

### **STEP 2: Create Codespace (1 minute)**

```
1. In your forked repo, click: Code (green button, top left)

2. Click: Codespaces tab

3. Click: Create codespace on main

4. WAIT 2-3 minutes for environment setup
   (VS Code will load in browser)
   
5. When ready, you'll see VS Code interface in browser
```

✅ **Codespace Ready!**

---

### **STEP 3: Upload Your Data (2 minutes)**

```
In Codespaces terminal (at bottom):

1. Type: cd /workspaces
   Press: Enter

2. Check files exist:
   Type: ls -la
   
   You should see your repo files
```

**Need to upload data files?**

```
If your data files aren't in the repo:
1. In VS Code Explorer (left side)
2. Right-click on "data" folder
3. Click: Upload files
4. Select: newssumm_clean.csv & news_summ_event_clustered.csv
5. WAIT for upload (~1-2 mins)
```

✅ **Data Ready!**

---

### **STEP 4: Open the Notebook (30 seconds)**

```
In VS Code Explorer (left):
1. Click: Train_All_11_Models_Google_Colab.ipynb

2. A preview opens at top

3. Wait for Jupyter to load (20 seconds)
```

✅ **Notebook Loaded!**

---

### **STEP 5: Select GPU Kernel (30 seconds)**

```
In notebook top right:
1. Click: Select Kernel

2. Choose: Python 3.10 (or latest available)
   (It will auto-use GPU if available)

3. CONFIRM it says "Ready"
```

✅ **Kernel Ready!**

---

### **STEP 6: Install Dependencies (2 minutes)**

```
In Codespaces terminal:

1. Copy-paste this command:

pip install -q torch transformers sentence-transformers bert-score rouge-score matplotlib seaborn pandas numpy scikit-learn tqdm

2. Press: Enter

3. WAIT for installation (~2 minutes)
```

✅ **Dependencies Installed!**

---

### **STEP 7: RUN ALL CELLS (2-4 hours) ⏱️**

```
In notebook:
1. Click: "Run All" button (top toolbar)
   OR
   Press: Ctrl+Shift+P
   Type: "Run All Cells"
   Press: Enter

2. WAIT - Training starts automatically!

3. You'll see cells execute one by one:
   ✓ Mount Drive (skipped in Codespaces)
   ✓ Check GPU
   ✓ Install libs
   ✓ Load data
   ✓ Configure 11 models
   ✓ Train all models (2-4 HOURS) ⏳
   ✓ Generate images
   ✓ Save results

4. Once done (after 2-4 hours):
   ✓ CSV file created
   ✓ 5 PNG images created
   ✓ Results saved to /workspaces/data/processed/
```

✅ **TRAINING STARTED!**

---

## ⏰ TIMELINE

```
NOW:           Step 1-7 (5-10 minutes total)
↓
Training:      2-4 hours (automatic)
               You can close browser, keep running
               Check back in 3 hours
↓
DONE:          Results ready in Codespace
               Download to your computer
```

---

## 📥 DOWNLOAD RESULTS (After Training)

```
In VS Code Explorer (left):
1. Expand: data → processed → 11_models_training_results

2. You'll see:
   ✓ all_11_models_comparison.csv
   ✓ 01_rouge_comparison.png
   ✓ 02_bertscore_faithfulness_comparison.png
   ✓ 03_error_metrics_comparison.png
   ✓ 04_metrics_heatmap.png
   ✓ 05_aims_improvement_analysis.png

3. Right-click each file:
   Click: Download
   
4. Save to your computer

5. DONE! ✅
```

---

## 🆘 IF SOMETHING GOES WRONG

### **"GPU not available"**
```
Solution:
1. Click: Select Kernel
2. Choose: Python 3.10 (explicit)
3. Re-run cell
```

### **"Out of memory error"**
```
Solution (in notebook cell 4):
Change: `[:30]` to `[:15]`
This reduces test clusters from 30 to 15
Then re-run
```

### **"Model download timeout"**
```
Solution:
1. Click: Restart kernel (at top)
2. Re-run cells
3. Downloading first model takes 10-15 mins (normal)
```

### **"Permission denied"**
```
Solution:
In terminal:
chmod +x -R /workspaces/
Then re-run
```

### **Connection dropped**
```
Solution:
1. Codespace auto-saves
2. Just refresh browser
3. Click: Resume
4. Training continues!
```

---

## 💡 PRO TIPS

### **While Training:**
- ✅ Keep browser tab open (or close, training continues)
- ✅ Check progress by scrolling cells
- ✅ Each cell shows completion time
- ✅ You can stop anytime (Ctrl+C in terminal)

### **Speed Up:**
- ✅ If T4 is slow, request A100 GPU
- ✅ Or run on local GPU if you have RTX card

### **Resume If Stopped:**
- ✅ Codespaces auto-saves progress
- ✅ Just re-open the Codespace
- ✅ Training resumes from last cell

---

## ✅ FULL CHECKLIST

Before you start:
- [ ] You have GitHub account
- [ ] You're ready to wait 2-4 hours
- [ ] Your Suvidha folder has the data CSVs

Ready? Let's go!

---

## 🎯 EXACT COMMANDS TO COPY-PASTE

### **Command 1: Install packages**
```
pip install -q torch transformers sentence-transformers bert-score rouge-score matplotlib seaborn pandas numpy scikit-learn tqdm
```

### **Command 2: Check GPU**
```
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### **Command 3: Run notebook directly** (alternative to "Run All")
```
jupyter nbconvert --to notebook --execute Train_All_11_Models_Google_Colab.ipynb --output results.ipynb
```

---

## 📊 EXPECTED OUTPUT

After 2-4 hours, in terminal you'll see:
```
✅ TRAINING & EVALUATION COMPLETE!
✅ Results saved to: /workspaces/data/processed/11_models_training_results/

Files generated:
  ✓ all_11_models_comparison.csv
  ✓ 01_rouge_comparison.png
  ✓ 02_bertscore_faithfulness_comparison.png
  ✓ 03_error_metrics_comparison.png
  ✓ 04_metrics_heatmap.png
  ✓ 05_aims_improvement_analysis.png

Final Rankings by ROUGE-1:
  1. AIMS ⭐          ROUGE-1=0.5200
  2. PEGASUS          ROUGE-1=0.4500
  3. LED              ROUGE-1=0.4700
  ... (all 11 models ranked)
```

---

## 🎉 DONE!

At this point:
✅ All 11 models trained
✅ All metrics computed
✅ All images generated
✅ CSV ready for analysis
✅ Results downloadable

**Take your CSV + 5 images to your paper!**

---

## ⏱️ TIMELINE SUMMARY

| Time | Action |
|------|--------|
| **Now** | Start these 7 steps |
| **5 min** | Codespace setup done |
| **7 min** | Dependencies installed |
| **7 min** | Notebook open |
| **~2-4 hrs** | Training (go get coffee ☕) |
| **Total: 2-4.5 hours** | Done! Results ready |

---

## 🚀 START NOW!

**GO TO:** https://github.com/LearnMernProjects/importance-aware-multi-document-summarization

**CLICK:** Fork

**THAT'S IT!** Follow the 7 steps above and you're good!

---

**Questions during setup?**
- Check the full guide: `VSCODE_GPU_OPTIONS.md`
- All troubleshooting is there

**Ready? LET'S GO!** ⚡🚀
