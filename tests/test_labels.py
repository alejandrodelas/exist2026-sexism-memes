"""The hard-label rules decide what the models learn, so they are tested explicitly."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import _hard_21, _hard_22, _hard_23, _sensors  # noqa: E402


def test_task21_needs_strict_majority():
    assert _hard_21(["YES"] * 4 + ["NO"] * 2) == "YES"
    assert _hard_21(["YES"] * 2 + ["NO"] * 4) == "NO"
    assert _hard_21(["YES"] * 3 + ["NO"] * 3) is None          # tie -> no hard label


def test_task22_maps_dash_to_no_and_ignores_unknown():
    assert _hard_22(["-", "-", "-", "DIRECT", "DIRECT", "UNKNOWN"]) == "NO"
    assert _hard_22(["JUDGEMENTAL"] * 3 + ["DIRECT"] * 2 + ["-"]) == "JUDGEMENTAL"
    assert _hard_22(["DIRECT", "DIRECT", "JUDGEMENTAL", "JUDGEMENTAL", "-", "-"]) is None


def test_task23_categories_need_two_votes_and_follow_task21():
    votes = [["OBJECTIFICATION"], ["OBJECTIFICATION", "SEXUAL-VIOLENCE"], ["STEREOTYPING-DOMINANCE"], "-", "-", "-"]
    assert _hard_23(votes, "YES") == ["OBJECTIFICATION"]
    assert _hard_23(votes, "NO") == ["NO"]
    assert _hard_23(votes, None) is None


def test_sensor_features_average_over_viewers():
    sens = {"users": ["A", "B"], "modalities": {"HR": {"by_user": {
        "A": {"garmin_hr_mean": 60.0}, "B": {"garmin_hr_mean": 70.0}}}}}
    f = _sensors(sens)
    assert f["HR__garmin_hr_mean__mean"] == 65.0
    assert f["n_viewers"] == 2
    assert np.isclose(f["HR__garmin_hr_mean__std"], 5.0)
