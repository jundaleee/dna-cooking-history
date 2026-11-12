"""J6: 양자화된 평평한 웰을 '오름'으로 읽지 않는다. 2·3차 판은 이미 분석한 원자료라 읽기만 한다."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rne.qpcr import controls as ct

ROOT = Path(__file__).resolve().parents[2]
PLATE2 = ROOT / "data/raw/qpcr/20261106_S104/분석A_ROX_수동기준선/2 hope 2v2_data.xls"


def _sigmoid(n=40, start=1.0, top=8.0, mid=24.0):
    c = np.arange(1, n + 1, dtype=float)
    return start + (top - start) / (1 + np.exp(-(c - mid) / 1.5))


def test_quantized_flat_well_is_not_rose():
    """평평하고 한 눈금만 움직이는 신호(차분 MAD 0): climb_snr는 커도 climb_rel이 작아 '오름'이 아니다."""
    s = np.ones(40)
    s[20] = 1.01  # 한 눈금
    f = ct.curve_features(s)
    assert f["climb_rel"] < 0.02
    t = pd.DataFrame([f])
    assert not bool(ct.rose_mask(t, dict(ct.DEFAULT_SETTINGS)).iloc[0])
    assert ct.DEFAULT_SETTINGS["climb_rel_min"] == 0.02


def test_real_amplification_still_rose():
    """S자 증폭(처음 1 → 고원 8)과 낮은 증폭(1 → 1.6)은 그대로 '오름'이다."""
    for top in (8.0, 1.6):
        f = ct.curve_features(_sigmoid(top=top) + np.random.default_rng(0).normal(0, 0.003, 40))
        assert f["climb_rel"] >= 0.02
        assert bool(ct.rose_mask(pd.DataFrame([f]), dict(ct.DEFAULT_SETTINGS)).iloc[0])


@pytest.mark.skipif(not PLATE2.exists(), reason="2차 판 원자료 없음")
def test_plate2_flat_well_a3_not_rose():
    """2차 판 A3는 '오름'이 아니고 나머지 95웰은 '오름'이다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = ct.well_signal_features(PLATE2)
    rose = ct.rose_mask(f, dict(ct.DEFAULT_SETTINGS))
    a3 = f["well"].astype(str).str.upper() == "A3"
    assert not bool(rose[a3].iloc[0]) and int(rose[~a3].sum()) == int((~a3).sum())
