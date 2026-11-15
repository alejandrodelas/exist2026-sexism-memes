"""Charts for the README from results/cv_results.csv and results/pyevall_icm.json."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
cv = pd.read_csv(ROOT / "results/cv_results.csv", dtype={"task": str})
icm = pd.DataFrame(json.load(open(ROOT / "results/pyevall_icm.json"))).rename(
    columns={"Normalized Information Contrast Model": "icm_norm", "Information Contrast model": "icm"})
cv = cv.merge(icm[["task", "model", "icm", "icm_norm"]], on=["task", "model"])
cv.round(4).to_csv(ROOT / "results/cv_results_with_icm.csv", index=False)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
titles = {"2.1": "2.1 · Is it sexist?", "2.2": "2.2 · Source intention", "2.3": "2.3 · Sexism category"}

# 1. every model, every task (macro-F1, 5-fold CV)
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
order = cv[cv.task == "2.1"].sort_values("macro_f1").model.tolist()
for ax, t in zip(axes, ["2.1", "2.2", "2.3"]):
    d = cv[cv.task == t].set_index("model").loc[order]
    best = d.macro_f1.idxmax()
    col = ["#d9822b" if m == best else "#5b8db8" if o else "#c9c9c9" for m, o in zip(d.index, d.original_choice)]
    ax.barh(d.index, d.macro_f1, color=col)
    for y, v in enumerate(d.macro_f1):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=7)
    ax.set_title(titles[t]); ax.set_xlabel("Macro-F1 (5-fold CV)"); ax.set_xlim(0, d.macro_f1.max() + 0.1)
fig.suptitle("All models, all tasks.  Orange = best now  ·  Blue = what I submitted in May 2026", y=0.99)
fig.tight_layout(); fig.savefig(FIG / "models_by_task.png"); plt.close(fig)

# 2. does adding the physiological sensors help?
pairs = [("word_char_tfidf_logreg", "word_char_sensor_logreg"),
         ("word_char_tfidf_logreg_balanced", "word_char_sensor_logreg_balanced")]
rows = []
for t in ["2.1", "2.2", "2.3"]:
    d = cv[cv.task == t].set_index("model").macro_f1
    rows.append({"task": titles[t], "Text only": d[pairs[0][0]], "Text + sensors": d[pairs[0][1]],
                 "Sensors only": d["sensor_extra_trees"]})
s = pd.DataFrame(rows).set_index("task")
s.round(4).to_csv(ROOT / "results/text_vs_sensors.csv")
ax = s.plot.bar(figsize=(8, 3.8), color=["#5b8db8", "#d9822b", "#c9c9c9"], rot=0, width=0.75)
ax.set_ylabel("Macro-F1 (5-fold CV)"); ax.set_xlabel("")
ax.set_title("Physiological signals add almost nothing; a small gain only for intention (2.2)")
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8); ax.set_ylim(0, 0.8)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "text_vs_sensors.png"); plt.close()

# 3. before vs after (submitted model vs best model now), macro-F1 and ICM-Norm
rows = []
for t in ["2.1", "2.2", "2.3"]:
    d = cv[cv.task == t]
    o, b = d[d.original_choice].iloc[0], d.sort_values("macro_f1", ascending=False).iloc[0]
    rows.append({"task": t, "submitted_model": o.model, "submitted_macro_f1": o.macro_f1, "submitted_icm_norm": o.icm_norm,
                 "best_model": b.model, "best_macro_f1": b.macro_f1, "best_icm_norm": b.icm_norm})
pd.DataFrame(rows).round(4).to_csv(ROOT / "results/before_vs_after.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
