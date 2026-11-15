"""
Load EXIST 2026 (sexism in memes) and turn the 6 annotators' votes into hard labels.

Hard-label rules (strict majority, as in the EXIST guidelines; ties are left out of training/evaluation):
  Task 2.1  sexist?           YES / NO   -> the class chosen by more than 3 of the 6 annotators
  Task 2.2  source intention  NO / DIRECT / JUDGEMENTAL -> the class chosen by more than 2 annotators
  Task 2.3  sexism category   multi-label -> NO if Task 2.1 is NO, otherwise every category chosen
                                             by more than 1 annotator
"""
from pathlib import Path
import json
from collections import Counter
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CATEGORIES = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
              "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]


def _hard_21(votes):
    c = Counter(votes)
    for k in ("YES", "NO"):
        if c[k] > 3:
            return k
    return None


def _hard_22(votes):
    c = Counter("NO" if v == "-" else v for v in votes if v != "UNKNOWN")
    best, n = c.most_common(1)[0]
    return best if n > 2 else None


def _hard_23(votes, label21):
    if label21 == "NO":
        return ["NO"]
    if label21 is None:
        return None
    c = Counter(cat for v in votes for cat in (v if isinstance(v, list) else [v]) if cat in CATEGORIES)
    cats = [k for k in CATEGORIES if c[k] > 1]
    return cats or None


def _sensors(sens):
    """Average each physiological feature over the people who watched the meme (+ spread between them)."""
    vals = {}
    for modality, block in sens.get("modalities", {}).items():
        for user, feats in block.get("by_user", {}).items():
            for f, v in feats.items():
                vals.setdefault(f"{modality}__{f}", []).append(np.nan if v is None else float(v))
    out = {}
    for k, v in vals.items():
        v = np.array(v, dtype=float)
        out[k + "__mean"] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        out[k + "__std"] = np.nanstd(v) if np.isfinite(v).sum() > 1 else np.nan
    out["n_viewers"] = len(sens.get("users", []))
    return out


def load(split="training"):
    path = DATA / ("EXIST2026_training.json" if split == "training" else "EXIST2026_test_clean.json")
    raw = json.load(open(path, encoding="utf-8"))
    rows, sens = [], []
    for k, x in raw.items():
        r = {"id": str(x["id_EXIST"]), "lang": x["lang"], "text": x["text"]}
        if split == "training":
            r["y21"] = _hard_21(x["labels_task2_1"])
            r["y22"] = _hard_22(x["labels_task2_2"])
            r["y23"] = _hard_23(x["labels_task2_3"], r["y21"])
            r["yes_share"] = np.mean([v == "YES" for v in x["labels_task2_1"]])
        rows.append(r)
        sens.append(_sensors(x.get("sensorial", {})))
    df = pd.DataFrame(rows)
    S = pd.DataFrame(sens)
    S.columns = [f"s__{c}" for c in S.columns]
    return pd.concat([df, S], axis=1)


if __name__ == "__main__":
    tr, te = load("training"), load("test")
    sc = [c for c in tr.columns if c.startswith("s__")]
    print(f"train {len(tr)} memes ({tr.lang.value_counts().to_dict()}), test {len(te)}; {len(sc)} sensor features")
    print("Task 2.1 hard labels:", tr.y21.value_counts(dropna=False).to_dict())
    print("Task 2.2 hard labels:", tr.y22.value_counts(dropna=False).to_dict())
    print("Task 2.3 hard labels:", Counter(c for v in tr.y23.dropna() for c in v))
