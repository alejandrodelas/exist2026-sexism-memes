"""
Official EXIST metrics (ICM-Hard, ICM-Hard normalised, F1) with PyEvALL on the cross-validated
(out-of-fold) predictions saved by run_experiments.py.

PyEvALL pins an old NumPy, so run it in its own environment:
    python -m venv .venv-pyevall && .venv-pyevall/bin/pip install pyevall
    .venv-pyevall/bin/python src/evaluate_pyevall.py
"""
from pathlib import Path
import contextlib, io, json, re
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "results/oof"
CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE",
        "MISOGYNY-NON-SEXUAL-VIOLENCE"]
HIER = {"2.2": {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}, "2.3": {"YES": CATS, "NO": []}}

rows = []
for pred in sorted(OOF.glob("pred_*.json")):
    task, model = re.match(r"pred_(\d\.\d)_(.+)\.json", pred.name).groups()
    params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}
    if task in HIER:
        params[PyEvALLUtils.PARAM_HIERARCHY] = HIER[task]
    with contextlib.redirect_stdout(io.StringIO()):
        rep = PyEvALLEvaluation().evaluate(str(pred), str(OOF / f"gold_{task}.json"),
                                           ["ICM", "ICMNorm", "FMeasure"], **params)
    out = {"task": task, "model": model}
    for m in rep.report["metrics"].values():
        val = m["results"]["average_per_test_case"] if "average_per_test_case" in m["results"] else None
        out[m["name"]] = round(val, 4) if isinstance(val, (int, float)) else val
    rows.append(out)
    print(out)
json.dump(rows, open(ROOT / "results/pyevall_icm.json", "w"), indent=1)

# ---- soft outputs (Task 2.1, learning with disagreement)
SOFT = ROOT / "results/oof_soft"
soft_rows = []
for pred in sorted(SOFT.glob("pred_soft_*.json")):
    model = pred.stem.replace("pred_soft_2.1_", "")
    with contextlib.redirect_stdout(io.StringIO()):
        rep = PyEvALLEvaluation().evaluate(str(pred), str(SOFT / "gold_soft_2.1.json"),
                                           ["ICMSoft", "ICMSoftNorm", "CrossEntropy"],
                                           **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED})
    out = {"task": "2.1-soft", "model": model}
    for m in rep.report["metrics"].values():
        val = m["results"].get("average_per_test_case")
        out[m["name"]] = round(val, 4) if isinstance(val, (int, float)) else val
    soft_rows.append(out)
    print(out)
json.dump(soft_rows, open(ROOT / "results/pyevall_soft.json", "w"), indent=1)
