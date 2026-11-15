"""
Learning with disagreement (soft labels) for Task 2.1.

Six people annotated every meme, and only 24 % of memes are unanimous. Instead of predicting only YES/NO,
EXIST also evaluates *soft* outputs: the share of people who would call the meme sexist.

Two ways to produce that probability, compared with 5-fold cross-validation on ALL 3,984 memes
(ties included):
  A. hard-trained: train on majority labels (ties dropped), output predict_proba
  B. soft-trained: every meme enters twice, as YES with weight = share of YES votes
     and as NO with weight = share of NO votes, so the model learns the disagreement itself
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from data import load
from models import zoo, SEED

ROOT = Path(__file__).resolve().parents[1]
OOF, RUNS = ROOT / "results/oof_soft", ROOT / "runs"
OOF.mkdir(parents=True, exist_ok=True)
NAME = "word_char_tfidf_logreg_balanced"

train, test = load("training"), load("test")
SENS = [c for c in train.columns if c.startswith("s__")]
base = zoo(SENS)[NAME]


def fit_hard(g):
    h = g[g.y21.notna()]
    return clone(base).fit(h, h.y21)


def fit_soft(g):
    both = pd.concat([g.assign(lbl="YES", w=g.yes_share), g.assign(lbl="NO", w=1 - g.yes_share)])
    both = both[both.w > 0]
    return clone(base).fit(both, both.lbl, model__sample_weight=both.w.values)


def p_yes(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index("YES")]


pred = {"hard_trained": pd.Series(index=train.index, dtype=float), "soft_trained": pd.Series(index=train.index, dtype=float)}
strat = (train.yes_share * 6).round().astype(int)            # stratify by number of YES votes
for lang, g in train.groupby("lang"):
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(g, strat.loc[g.index]):
        gt, gv = g.iloc[tr], g.iloc[va]
        pred["hard_trained"].loc[gv.index] = p_yes(fit_hard(gt), gv)
        pred["soft_trained"].loc[gv.index] = p_yes(fit_soft(gt), gv)

y = train.yes_share
rows = []
for k, p in pred.items():
    pc = p.clip(1e-6, 1 - 1e-6)
    ce = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
    rows.append({"model": k, "mean_abs_error_vs_vote_share": (p - y).abs().mean(), "cross_entropy": ce,
                 "corr_with_vote_share": np.corrcoef(p, y)[0, 1]})
    json.dump([{"id": i, "value": {"YES": round(float(v), 4), "NO": round(1 - float(v), 4)}, "test_case": "EXIST2026"}
               for i, v in zip(train.id, p)], open(OOF / f"pred_soft_2.1_{k}.json", "w"))
json.dump([{"id": i, "value": {"YES": round(float(v), 4), "NO": round(1 - float(v), 4)}, "test_case": "EXIST2026"}
           for i, v in zip(train.id, y)], open(OOF / "gold_soft_2.1.json", "w"))
res = pd.DataFrame(rows).round(4)
res.to_csv(ROOT / "results/soft_labels_2_1.csv", index=False)
print(res.to_string(index=False))

# soft run for the test set with the soft-trained model
out = pd.Series(index=test.index, dtype=float)
for lang in ("en", "es"):
    out.loc[test.lang == lang] = p_yes(fit_soft(train[train.lang == lang]), test[test.lang == lang])
json.dump([{"id": i, "value": {"YES": round(float(v), 4), "NO": round(1 - float(v), 4)}, "test_case": "EXIST2026"}
           for i, v in zip(test.id, out)], open(RUNS / "task2_1_soft_Alejandro_1.json", "w"), indent=1)
print("soft run written")
