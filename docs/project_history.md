# Project history

## Context
An individual project ("Team Alejandro") for the course *Natural Language and Information Retrieval* (BSc Data Science, UPV, spring 2026), built around the EXIST 2026 shared task at CLEF. I did all of it myself: data handling, every model, the runs and the presentation.

| When | What |
|---|---|
| Spring 2026 | Course labs on the EXIST data: preprocessing, static and contextual embeddings, **fine-tuning transformers (RoBERTa) with SHAP/LIME explanations**, and **LLM prompting** (zero- and few-shot with FLAN-T5). See `notebooks/02_…` and `notebooks/03_…` |
| May 2026 | Model zoo with TF-IDF, sensor features and linear and tree models; an internal 80/20 split per language |
| 14 May 2026 | **9 official runs submitted** (3 per subtask, hard labels) → `original_submission/` |
| 27 May 2026 | Course presentation, *Model Selection for Sexism Identification in Memes* |
| Result | **43rd of 185 official runs (top 25 %)** in subtask 2.3, sexism categorisation |
| Afterwards | This rebuild (below) |

## Why the advanced models were not submitted in May
Transformers and image encoders needed GPU time I didn't have before the deadline. A run has to cover all 1,053 test memes without failing, and I didn't want to submit a model chosen for being sophisticated rather than for validated performance. So the final runs used the configurations that were fully validated.

## What the rebuild changed

| May 2026 | Rebuild |
|---|---|
| Single 80/20 split | **5-fold stratified cross-validation**, per language |
| Custom selection score | Macro-F1 for selection, plus the **official ICM metric** (PyEvALL) on out-of-fold predictions |
| No class weights | **Class balancing**: the biggest single gain, on rare classes such as JUDGEMENTAL |
| Hard labels only | Also a **soft run** for 2.1 (learning with disagreement), trained on vote shares |
| Scripts in notebooks | Python package layout, unit tests for the label rules, run validator, CI |

Same data, same model families, same idea of comparing text-only, sensor-only and combined models. The axis of the original work stays; it's just measured more carefully.
