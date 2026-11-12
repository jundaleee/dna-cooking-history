"""templates/ 양식 시험(열 이름과 칸 구성)."""
from pathlib import Path

import pandas as pd

from rne.common.schema import SAMPLE_MAP_COLUMNS
from rne.qpcr.results import load_target_alias

T = Path(__file__).resolve().parents[2] / "templates"


def test_sample_map_template_columns():
    d = pd.read_csv(T / "sample_map_template.csv")
    assert list(d.columns) == SAMPLE_MAP_COLUMNS and len(d) == 5
    assert {"marinade", "commercial"} <= set(d["group"])


def test_standard_curve_layout_template():
    d = pd.read_csv(T / "standard_curve_layout_template.csv")
    assert list(d.columns) == ["well", "sample_id", "target", "role"]
    assert len(d) == 96 and d["well"].is_unique
    assert set(d["role"]) == {"std", "ntc", "unknown"}
    assert (d[d.role == "ntc"].groupby("target").size() >= 2).all()
    assert (d[d.role == "std"].groupby("target")["sample_id"].nunique() == 5).all()


def test_alias_json_targets():
    a = load_target_alias(T / "target_alias.json")
    assert set(a.values()) == {"S104", "L586"}


def test_plate96_example_matches_controls_format():
    d = pd.read_csv(T / "plate96_layout_example.csv", encoding="utf-8-sig")
    assert {"well", "target", "kind"} <= set(d.columns) and len(d) == 96 and d["well"].is_unique
    assert d.kind.value_counts().to_dict() == {"sample": 80, "ntc": 8, "noprimer": 4, "mixonly": 2, "water": 2}
