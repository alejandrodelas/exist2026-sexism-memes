"""Check every run in runs/ has the PyEvALL format: 1,053 unique ids, valid labels, test_case EXIST2026."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
VALID = {"2_1": {"YES", "NO"}, "2_2": {"NO", "DIRECT", "JUDGEMENTAL"},
         "2_3": {"NO", "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
                 "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"}}
ok = True
for f in sorted((ROOT / "runs").glob("task*.json")):
    run = json.load(open(f))
    task = f.name[4:7]
    ids = [r["id"] for r in run]
    problems = []
    if len(run) != 1053 or len(set(ids)) != 1053:
        problems.append(f"{len(run)} rows / {len(set(ids))} unique ids")
    if any(r["test_case"] != "EXIST2026" for r in run):
        problems.append("wrong test_case")
    for r in run:
        v = r["value"]
        if isinstance(v, dict):                                   # soft run
            if set(v) - VALID[task] or abs(sum(v.values()) - 1) > 1e-3:
                problems.append(f"bad soft value {v}"); break
        elif isinstance(v, list):
            if not v or set(v) - VALID[task] or ("NO" in v and len(v) > 1):
                problems.append(f"bad labels {v}"); break
        elif v not in VALID[task]:
            problems.append(f"bad label {v}"); break
    print(f"{'OK ' if not problems else 'BAD'} {f.name} {'; '.join(problems)}")
    ok &= not problems
sys.exit(0 if ok else 1)
