"""원시 곡선 내보내기 읽기 시험. 모든 파일은 가짜(fake_)이며 형식 문서화용이다."""
from pathlib import Path

import numpy as np
import pytest

from rne.qpcr import read_amplification, read_melt, read_multicomponent
from rne.qpcr.raw import AMPLIFICATION_COLUMNS, MELT_COLUMNS, MULTICOMPONENT_COLUMNS

FIX = Path(__file__).parent / "fixtures"
ALIAS = {"COI-104": "S104", "COI-586": "L586"}
META = "* Experiment File Name = C:\\x\\FAKE.eds\n* Passive Reference = ROX\n\n"


def test_amplification_from_sections_file():
    df = read_amplification(FIX / "fake_results_sections.txt", ALIAS)
    assert list(df.columns) == AMPLIFICATION_COLUMNS
    assert len(df) == 3
    assert (df["target"] == "S104").all()
    assert df["cycle"].tolist() == [1.0, 2.0, 3.0]
    assert df["delta_rn"].iloc[-1] == pytest.approx(0.003)


def test_amplification_csv_numeric_well(tmp_path):
    p = tmp_path / "fake_amp.csv"
    p.write_text(META.replace("\n\n", "\n") + "Well,Well Position,Cycle,Target Name,Rn,Delta Rn\n"
                 "13,B1,2,long,1.1,0.1\n13,B1,1,long,1.0,0.0\n", encoding="utf-8")
    df = read_amplification(p, ALIAS | {"long": "L586"})
    assert df["well"].tolist() == ["B1", "B1"]
    assert df["cycle"].tolist() == [1.0, 2.0]  # 사이클 순으로 정렬
    assert (df["target"] == "L586").all()


def test_multicomponent_wide_to_long(tmp_path):
    p = tmp_path / "fake_mc.txt"
    lines = [META + "[Multicomponent Data]", "Well\tCycle\tSYBR\tROX"]
    for w in ("A1", "A2"):
        for c in range(1, 4):
            lines.append(f"{w}\t{c}\t{100 + c}\t{5000 + c}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = read_multicomponent(p)
    assert list(df.columns) == MULTICOMPONENT_COLUMNS
    assert set(df["dye"]) == {"SYBR", "ROX"}
    assert len(df) == 12
    rox = df[(df.dye == "ROX") & (df.well == "A2")]
    assert rox["fluorescence"].tolist() == [5001.0, 5002.0, 5003.0]


def test_multicomponent_no_dyes(tmp_path):
    p = tmp_path / "fake_mc_empty.txt"
    p.write_text("Well\tCycle\tROX\nA1\t1\t\n", encoding="utf-8")
    df = read_multicomponent(p)
    assert df.empty and list(df.columns) == MULTICOMPONENT_COLUMNS


def test_melt_with_and_without_derivative(tmp_path):
    temps = np.arange(60.0, 95.0, 0.5)
    fluor = 1.0 / (1.0 + np.exp((temps - 80.0) / 0.8))
    p1 = tmp_path / "fake_melt.txt"
    rows = ["Well\tWell Position\tReading\tTemperature\tFluorescence\tDerivative"]
    for i, (t, f) in enumerate(zip(temps, fluor)):
        rows.append(f"1\tA1\t{i + 1}\t{t}\t{f}\t{0.5}")
    p1.write_text(META + "[Melt Curve Raw Data]\n" + "\n".join(rows) + "\n", encoding="utf-8")
    d1 = read_melt(p1)
    assert list(d1.columns) == MELT_COLUMNS
    assert (d1["derivative"] == 0.5).all()

    p2 = tmp_path / "fake_melt_noderiv.txt"
    rows = ["Well\tTemperature\tFluorescence"] + [f"A1\t{t}\t{f}" for t, f in zip(temps, fluor)]
    p2.write_text("\n".join(rows) + "\n", encoding="utf-8")
    d2 = read_melt(p2)
    peak_T = d2.loc[d2["derivative"].idxmax(), "temperature_C"]
    assert abs(peak_T - 80.0) <= 0.5  # −dF/dT 봉우리가 Tm


def test_wrong_section_raises():
    with pytest.raises(ValueError):
        read_multicomponent(FIX / "fake_results_sections.txt")


REAL = Path(__file__).resolve().parents[2] / "data" / "raw" / "qpcr" / "20261106_S104" / "분석A_ROX_수동기준선" / "2 hope 2v2_data.xls"


@pytest.mark.skipif(not REAL.exists(), reason="2차 판 실측 내보내기 없음")
def test_real_export_readers():
    # 실제 내보내기가 있으면 그것으로도 돈다(읽기만)
    import warnings

    from rne.qpcr.results import passive_reference

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert passive_reference(REAL) == "ROX"  # `*` 없는 두 칸 메타데이터
        mc = read_multicomponent(REAL)
        assert mc["cycle"].max() == 40  # 멜트 읽기(41–157)는 잘라냄
        assert len(read_multicomponent(REAL, max_cycle=None)) > len(mc)
        me = read_melt(REAL)  # 넓은 Melt Region 시트
        assert me["well"].nunique() == 96 and me["derivative"].notna().all()
        assert 60.0 < me["temperature_C"].min() < 61.0 and 94.0 < me["temperature_C"].max() < 96.0
        a = read_amplification(REAL, {"Target 1": "S104"})
        assert a["well"].nunique() == 96 and a["delta_rn"].notna().all()
