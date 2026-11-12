"""read_steponeplus 시험. 픽스처는 모두 가짜(fake_)이며 형식 문서화용이다."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rne.common.schema import QPCR_LONG_COLUMNS, check_columns
from rne.qpcr import read_steponeplus
from rne.qpcr._io import is_ct_header, normalize_well, read_metadata
from rne.qpcr.results import load_target_alias, map_target, passive_reference

FIX = Path(__file__).parent / "fixtures"
ALIAS = {"COI-104": "S104", "COI-586": "L586"}
FORMATS = ["fake_results_v23.xls", "fake_results_nometa.txt"]


@pytest.mark.parametrize("name", FORMATS)
def test_read_formats_same_content(name):
    with pytest.warns(UserWarning, match="mystery-target"):
        df = read_steponeplus(FIX / name, ALIAS)
    check_columns(df, QPCR_LONG_COLUMNS, "qpcr_long")
    assert len(df) == 10
    assert df["well"].tolist()[:4] == ["A1", "A2", "A3", "A4"]
    a4 = df.set_index("well").loc["A4"]
    assert a4["undetermined"] and np.isnan(a4["ct"])
    assert a4["instrument_flags"] == "NOAMP"
    c1 = df.set_index("well").loc["C1"]
    assert c1["instrument_flags"] == "BADROX;EXPFAIL"
    assert c1["melt_peaks"] == 2
    assert df.set_index("well").loc["B1", "std_quantity"] == 10000
    assert np.isnan(df.set_index("well").loc["A1", "std_quantity"])
    # 모르는 표적은 그대로
    assert df.set_index("well").loc["C2", "target"] == "mystery-target"
    # 기술 반복 번호
    reps = df[(df.sample_id == "C-080-30-1") & (df.target == "S104")]["tech_rep"].tolist()
    assert reps == [1, 2]


def test_both_formats_identical_core():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = read_steponeplus(FIX / FORMATS[0], ALIAS)
        b = read_steponeplus(FIX / FORMATS[1], ALIAS)
    cols = ["well", "sample_id", "target", "ct", "undetermined", "tech_rep", "instrument_flags", "task"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_sections_file_and_metadata():
    df = read_steponeplus(FIX / "fake_results_sections.txt", ALIAS)
    assert len(df) == 4
    assert df.loc[0, "plate_id"] == "FAKE_plate"
    assert np.isnan(df["melt_peaks"]).all()  # Tm 열 없음
    assert passive_reference(FIX / "fake_results_sections.txt") == "ROX"
    assert passive_reference(FIX / "fake_results_nometa.txt") is None
    meta = read_metadata(FIX / "fake_results_v23.xls")
    assert meta["Chemistry"] == "SYBR_GREEN"


def test_plate_id_override():
    df = read_steponeplus(FIX / "fake_results_sections.txt", ALIAS, plate_id="P1")
    assert (df["plate_id"] == "P1").all()


def test_ct_header_variants():
    for h in ["CT", "Ct", "Cт", "CÑ‚", "Cq"]:
        assert is_ct_header(h), h
    for h in ["Ct Mean", "Cт SD", "Ct Threshold", "Comments", "Quantity"]:
        assert not is_ct_header(h), h


def test_normalize_well():
    assert normalize_well("a01") == "A1"
    assert normalize_well(13) == "B1"
    assert normalize_well("96") == "H12"
    assert normalize_well(1.0) == "A1"


def test_alias_case_insensitive_and_default():
    assert map_target("coi-104", ALIAS) == "S104"
    assert map_target("S104", {}) == "S104"
    alias = load_target_alias()  # data/templates 대응표 또는 내장표
    assert map_target("short", alias) == "S104"
    assert map_target("586 bp", alias) == "L586"


def test_control_target_name_passes_without_warning():
    """대조 칸 표적 CTRL은 경고 없이 그대로. 모르는 이름은 여전히 경고한다."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert map_target("CTRL", {}) == "CTRL"
        assert map_target(" ctrl ", {}) == "CTRL"
    with pytest.warns(UserWarning):
        assert map_target("mystery", {}) == "mystery"


def test_missing_section_raises(tmp_path):
    p = tmp_path / "fake_bad.txt"
    p.write_text("[Amplification Data]\nWell\tCycle\nA1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_steponeplus(p, ALIAS)
