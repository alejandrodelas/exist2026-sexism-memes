"""Exploratory analysis of the EXIST 2026 meme data: labels, annotator disagreement, text, sensors."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
raw = json.load(open(ROOT / "data/EXIST2026_training.json", encoding="utf-8"))
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

rows = []
for x in raw.values():
    v = x["labels_task2_1"]
    rows.append({"lang": x["lang"], "yes_votes": sum(l == "YES" for l in v), "n_ann": len(v),
                 "words": len(x["text"].split()), "chars": len(x["text"]),
                 "has_sensors": bool(x.get("sensorial", {}).get("modalities"))})
d = pd.DataFrame(rows)

stats = {
    "memes": len(d), "by_language": d.lang.value_counts().to_dict(),
    "annotators_per_meme": int(d.n_ann.median()),
    "unanimous_share": float(((d.yes_votes == 0) | (d.yes_votes == 6)).mean()),
    "tie_3_3_share": float((d.yes_votes == 3).mean()),
    "median_words": float(d.words.median()),
    "memes_with_sensor_data": float(d.has_sensors.mean()),
}
json.dump(stats, open(ROOT / "results/eda_stats.json", "w"), indent=2)
print(stats)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
for lang, col, off in (("en", "#5b8db8", -0.2), ("es", "#d9822b", 0.2)):
    c = d[d.lang == lang].yes_votes.value_counts(normalize=True).reindex(range(7), fill_value=0)
    ax.bar(np.arange(7) + off, c.values, 0.4, color=col, label={"en": "English", "es": "Spanish"}[lang])
ax.axvspan(2.5, 3.5, color="#eee", zorder=0)
ax.text(3, ax.get_ylim()[1] * 0.92, "3-3 tie\n(no hard label)", ha="center", fontsize=8)
ax.set_xticks(range(7)); ax.set_xlabel("Annotators (out of 6) who said the meme is sexist")
ax.set_ylabel("Share of memes"); ax.legend(frameon=False)
ax.set_title(f"People disagree: only {stats['unanimous_share']:.0%} of memes are unanimous")

ax = axes[1]
ax.hist([d[d.lang == "en"].words.clip(upper=80), d[d.lang == "es"].words.clip(upper=80)], bins=20,
        color=["#5b8db8", "#d9822b"], label=["English", "Spanish"])
ax.set_xlabel("Words of OCR text per meme (capped at 80)"); ax.set_ylabel("Memes"); ax.legend(frameon=False)
ax.set_title(f"Very short texts: median {stats['median_words']:.0f} words")
fig.tight_layout(); fig.savefig(FIG / "eda_disagreement_and_length.png"); plt.close(fig)
