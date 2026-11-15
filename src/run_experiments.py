"""
Re-run EXIST 2026 Tasks 2.1, 2.2 and 2.3 end to end.

1. 5-fold stratified cross-validation (separate English / Spanish models) for every model in the zoo
2. pick the 3 best configurations per task by macro-F1
3. retrain them on all training memes and write 9 PyEvALL-format runs for the 1,053 test memes

Out-of-fold predictions are saved so the official metric (ICM-Hard) can be computed with PyEvALL.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from data import load, CATEGORIES
from models import zoo, ORIGINAL_CHOICE, SEED

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES, RUNS = ROOT / "results", ROOT / "runs"
RES.mkdir(exist_ok=True); RUNS.mkdir(exist_ok=True)
TEST_CASE = "EXIST2026"
K = 5

train, test = load("training"), load("test")
SENS = [c for c in train.columns if c.startswith("s__")]
ZOO = zoo(SENS)
LABELS23 = ["NO"] + CATEGORIES
mlb = MultiLabelBinarizer(classes=LABELS23)


def scores(task, y, p):
    if task == "2.1":
        return {"macro_f1": f1_score(y, p, average="macro"), "f1_yes": f1_score(y, p, pos_label="YES")}
    if task == "2.2":
        return {"macro_f1": f1_score(y, p, average="macro"),
                **{f"f1_{c.lower()}": f for c, f in zip(["NO", "DIRECT", "JUDGEMENTAL"],
                   f1_score(y, p, labels=["NO", "DIRECT", "JUDGEMENTAL"], average=None))}}
    Y, P = mlb.fit_transform(y), mlb.transform(p)
    return {"macro_f1": f1_score(Y, P, average="macro", zero_division=0),
            "micro_f1": f1_score(Y, P, average="micro", zero_division=0)}


def oof(name, d, target):
    """out-of-fold predictions for a single-label task, one model per language"""
    pred = pd.Series(index=d.index, dtype=object)
    for lang, g in d.groupby("lang"):
        for tr, va in StratifiedKFold(K, shuffle=True, random_state=SEED).split(g, g[target]):
            m = clone(ZOO[name]).fit(g.iloc[tr], g[target].iloc[tr])
            pred.loc[g.index[va]] = m.predict(g.iloc[va])
    return pred


def categories(model, X):
    """multi-label prediction; every sexist meme gets at least its most likely category"""
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X); thr = 0.5
    else:
        s = model.decision_function(X); thr = 0.0
    out = []
    for row in s:
        idx = np.where(row >= thr)[0]
        if len(idx) == 0:
            idx = [int(np.argmax(row))]
        out.append([CATEGORIES[i] for i in idx])
    return out


def fit_categories(name, g):
    sx = g[g.y23.apply(lambda v: v != ["NO"])]
    Y = MultiLabelBinarizer(classes=CATEGORIES).fit_transform(sx.y23)
    return OneVsRestClassifier(clone(ZOO[name])).fit(sx, Y)


def oof23(name, d, gate):
    pred = pd.Series(index=d.index, dtype=object)
    strat = d.y23.apply(lambda v: "NO" if v == ["NO"] else v[0])
    for lang, g in d.groupby("lang"):
        for tr, va in StratifiedKFold(K, shuffle=True, random_state=SEED).split(g, strat.loc[g.index]):
            m = fit_categories(name, g.iloc[tr])
            v = g.iloc[va]
            cats = categories(m, v)
            pred.loc[v.index] = [["NO"] if gate.loc[i] == "NO" else c for i, c in zip(v.index, cats)]
    return pred


log, store = [], {}
t0 = time.time()

# ---------------------------------------------------------------- Task 2.1
d21 = train[train.y21.notna()].reset_index(drop=True)
for name in ZOO:
    p = oof(name, d21, "y21"); store[("2.1", name)] = p
    log.append({"task": "2.1", "model": name, **scores("2.1", d21.y21, p)})
    print(f"[{time.time()-t0:5.0f}s] 2.1 {name:36s} macroF1={log[-1]['macro_f1']:.4f}")

# ---------------------------------------------------------------- Task 2.2
d22 = train[train.y22.notna()].reset_index(drop=True)
for name in ZOO:
    p = oof(name, d22, "y22"); store[("2.2", name)] = p
    log.append({"task": "2.2", "model": name, **scores("2.2", d22.y22, p)})
    print(f"[{time.time()-t0:5.0f}s] 2.2 {name:36s} macroF1={log[-1]['macro_f1']:.4f}")

# ---------------------------------------------------------------- Task 2.3 (gated by the best Task 2.1 model)
res = pd.DataFrame(log)
best21 = res[res.task == "2.1"].sort_values("macro_f1", ascending=False).model.iloc[0]
gate_all = pd.Series(index=train.index, dtype=object)
# Task 2.1 out-of-fold decisions for every meme with a 2.3 label
d23 = train[train.y23.notna()].copy()
gate_all.loc[d23.index] = oof(best21, d23.assign(g=d23.y23.apply(lambda v: "NO" if v == ["NO"] else "YES")), "g")
d23 = d23.reset_index(drop=True); gate = gate_all.dropna().reset_index(drop=True)
for name in ZOO:
    p = oof23(name, d23, gate); store[("2.3", name)] = p
    log.append({"task": "2.3", "model": name, **scores("2.3", d23.y23, p)})
    print(f"[{time.time()-t0:5.0f}s] 2.3 {name:36s} macroF1={log[-1]['macro_f1']:.4f}")

res = pd.DataFrame(log)
res["original_choice"] = res.apply(lambda r: ORIGINAL_CHOICE[r.task] == r.model, axis=1)
res = res.sort_values(["task", "macro_f1"], ascending=[True, False])
res.round(4).to_csv(RES / "cv_results.csv", index=False)
print(res.round(4).to_string(index=False))

# out-of-fold predictions in PyEvALL format (for the official ICM-Hard metric)
oofdir = RES / "oof"; oofdir.mkdir(exist_ok=True)
gold = {"2.1": d21[["id", "y21"]], "2.2": d22[["id", "y22"]], "2.3": d23[["id", "y23"]]}
for task, g in gold.items():
    json.dump([{"id": i, "value": v, "test_case": TEST_CASE} for i, v in g.values],
              open(oofdir / f"gold_{task}.json", "w"))
for (task, name), p in store.items():
    ids = gold[task].id.values
    json.dump([{"id": i, "value": v, "test_case": TEST_CASE} for i, v in zip(ids, p.values)],
              open(oofdir / f"pred_{task}_{name}.json", "w"))

# ---------------------------------------------------------------- final runs on the test set
top = {t: res[res.task == t].model.head(3).tolist() for t in ("2.1", "2.2", "2.3")}
json.dump({"best_task21_gate": best21, "top3": top}, open(RES / "selected_models.json", "w"), indent=2)


def fit_predict(task, name):
    out = pd.Series(index=test.index, dtype=object)
    for lang in ("en", "es"):
        te = test[test.lang == lang]
        if task == "2.1":
            g = d21[d21.lang == lang]; out.loc[te.index] = clone(ZOO[name]).fit(g, g.y21).predict(te)
        elif task == "2.2":
            g = d22[d22.lang == lang]; out.loc[te.index] = clone(ZOO[name]).fit(g, g.y22).predict(te)
        else:
            g = d21[d21.lang == lang]; gate_te = clone(ZOO[best21]).fit(g, g.y21).predict(te)
            m = fit_categories(name, d23[d23.lang == lang])
            out.loc[te.index] = [["NO"] if z == "NO" else c for z, c in zip(gate_te, categories(m, te))]
    return out


for task, names in top.items():
    for k, name in enumerate(names, 1):
        p = fit_predict(task, name)
        run = [{"id": i, "value": v, "test_case": TEST_CASE} for i, v in zip(test.id, p)]
        fn = RUNS / f"task{task.replace('.', '_')}_hard_Alejandro_{k}.json"
        json.dump(run, open(fn, "w"), indent=1, ensure_ascii=False)
        print(f"run {fn.name}: {name}")
print(f"done in {time.time()-t0:.0f}s")
