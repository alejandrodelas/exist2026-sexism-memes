"""Per-language results and confusion matrices for the best model of Tasks 2.1 and 2.2 (out-of-fold)."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

from data import load

ROOT = Path(__file__).resolve().parents[1]
OOF, FIG = ROOT / "results/oof", ROOT / "results/figures"
lang = load("training").set_index("id").lang
sel = json.load(open(ROOT / "results/selected_models.json"))["top3"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def frame(task, model):
    g = pd.DataFrame(json.load(open(OOF / f"gold_{task}.json"))).rename(columns={"value": "gold"})
    p = pd.DataFrame(json.load(open(OOF / f"pred_{task}_{model}.json"))).rename(columns={"value": "pred"})
    d = g[["id", "gold"]].merge(p[["id", "pred"]], on="id")
    d["lang"] = d.id.map(lang)
    return d


rows = []
for task in ("2.1", "2.2"):
    for model in sel[task]:
        d = frame(task, model)
        for lg, x in d.groupby("lang"):
            rows.append({"task": task, "model": model, "lang": lg, "macro_f1": f1_score(x.gold, x.pred, average="macro")})
per_lang = pd.DataFrame(rows).round(4)
per_lang.to_csv(ROOT / "results/per_language.csv", index=False)
print(per_lang.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, task, labels in ((axes[0], "2.1", ["NO", "YES"]), (axes[1], "2.2", ["NO", "DIRECT", "JUDGEMENTAL"])):
    d = frame(task, sel[task][0])
    cm = confusion_matrix(d.gold, d.pred, labels=labels, normalize="true")
    ax.imshow(cm, cmap="Oranges", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", color="white" if cm[i, j] > 0.6 else "#222")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Majority of annotators")
    ax.set_title(f"Task {task}: {sel[task][0]}", fontsize=8)
fig.suptitle("Where the best models go wrong (rows sum to 100 %)")
fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png"); plt.close(fig)
