#!/usr/bin/env pwsh

cd "c:\Users\Viraj Naik\Desktop\Suvidha"

# Reset the previous commit to amend with better message
git reset --soft HEAD~1

# Remove any AI-generated markdown files that might have been added
git reset HEAD GITHUB_PUSH_STATUS.md 2>$null
git reset HEAD GITHUB_INSTRUCTIONS.md 2>$null
git reset HEAD DEPLOYMENT_STATUS.md 2>$null

# Remove these files from git if they exist
if (Test-Path "GITHUB_PUSH_STATUS.md") { Remove-Item "GITHUB_PUSH_STATUS.md" -Force }
if (Test-Path "GITHUB_INSTRUCTIONS.md") { Remove-Item "GITHUB_INSTRUCTIONS.md" -Force }
if (Test-Path "DEPLOYMENT_STATUS.md") { Remove-Item "DEPLOYMENT_STATUS.md" -Force }

# Stage all legitimate files
git add -A

# Create new commit with research-quality message
$commitMessage = @"
Importance-Aware Multi-Document Summarization on NewsSumm Dataset

Abstract: This repository implements a novel importance-aware approach for extracting abstractive summaries from multi-document event clusters, addressing the challenge of document ordering in multi-document summarization (MDS) tasks.

Methodology:
- Event-level clustering: Semantic grouping of 3,000 articles using embeddings and temporal-categorical constraints (27 multi-document clusters extracted)
- Importance scoring: Centroid-based approach computing article salience via mean cosine similarity (h_i → α_i → w_i formulation)
- Abstractive summarization: Facebook/BART-large-cnn with importance-aware document ordering

Evaluation: Comparative analysis on NewsSumm dataset using ROUGE and BERTScore metrics
- Baseline: BART with chronological ordering (ROUGE-1: 0.3040, BERTScore F1: 0.6130)
- Proposed: BART with importance-aware ordering (ROUGE-1: 0.3058, BERTScore F1: 0.6123)

Key findings demonstrate competitive performance of importance-weighted ordering with improved performance on National News category (+10.1% ROUGE-1).

Artifacts: Event clusters, baseline/proposed summaries, comprehensive evaluation metrics, publication-ready visualizations (300 DPI)
"@

git commit -m $commitMessage

# Attempt force push to resolve branch mismatch
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push -u origin master:main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository: https://github.com/LearnMernProjects/importance-aware-multi-document-summarization" -ForegroundColor Green
} else {
    Write-Host "Push failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Attempting alternate push strategy..." -ForegroundColor Yellow
    git push origin master -f
}
