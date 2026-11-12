"""문헌 환산 재현. 80°C 60분 DDI 증가, E = 1, ΔL = 482 bp.

공통 식: ΔDDI = 482 · k(T_src) · [k(80)/k(T_src)] · 60 / ln 2, 아레니우스 환산.
"""
import pytest

from rne.models import literature as lit
from rne.models import rate_ratio


def test_openwetware():
    """문헌 속도 정리: 37°C 3×10⁻¹¹/s, 원 조건의 Ea 121로 80°C 환산.

    k(37) = 1.8×10⁻⁹ /min, 80/37 속도 비(Ea 121) ≈ 302.7 → k(80) ≈ 5.45×10⁻⁷ /min,
    ΔDDI = 482·5.45×10⁻⁷·60/ln2 ≈ 0.0228.
    표 값 0.02는 유효숫자 1자리라 상대 5% 기준은 맞지 않는다(0.0228은 0.02보다 14% 큼).
    표시 자릿수 기준(반올림하면 0.02)으로 확인하고, 차이는 리포트에 적는다.
    Ea 130을 쓰면 약 0.035가 된다.
    """
    pred = lit.literature_predictions()["openwetware"]
    assert pred == pytest.approx(0.0228, rel=0.01)
    assert round(pred, 2) == 0.02


def test_hird_boil():
    """Hird 2006 끓임: 100°C 32분, 351·81 bp 차이 변화 ≤ 1 cycle.

    Δλ ≤ 1·ln2/270 (32분 동안) → ×(60/32) 시간 비 → ×k(80)/k(100) = 0.0932 (Ea 130)
    → ΔDDI ≤ (482/270)·1·1.875·0.0932 ≈ 0.312.
    """
    assert lit.literature_predictions()["hird_boil_max"] == pytest.approx(0.31, rel=0.05)


def test_hird_autoclave():
    """Hird 2006 오토클레이브: 121°C 15분, 81 bp +1.41, 351 bp +10 → 차이 8.59 cycle.

    ΔDDI = (482/270)·8.59·(60/15)·k(80)/k(121) = 1.785·8.59·4·0.0100 ≈ 0.613.
    """
    assert lit.literature_predictions()["hird_autoclave"] == pytest.approx(0.61, rel=0.05)


def test_brisco():
    """Brisco 2010: 99°C 0·20·30분, 200 bp 증폭 가능 분율 0.93·0.48·0.32.

    λ(t) = −ln(AF)/200 = 3.63×10⁻⁴, 3.67×10⁻³, 5.70×10⁻³. 세 점 최소제곱 기울기
    k(99) = 1.760×10⁻⁴ /min (0분 값이 λ₀를 흡수). k(80)/k(99) = 0.1043 (Ea 130)
    → ΔDDI = 482·1.760×10⁻⁴·0.1043·60/ln2 ≈ 0.766. Ea 121이면 비가 0.1219 → 0.896.
    """
    assert lit.brisco_rate_per_min() == pytest.approx(1.760e-4, rel=0.005)
    assert lit.literature_predictions()["brisco"] == pytest.approx(0.77, rel=0.05)
    assert lit.literature_predictions(Ea_kJ=121.0)["brisco"] == pytest.approx(0.90, rel=0.05)


def test_hird_inconsistency_in_hird_units():
    """01 문서 5-1: 오토클레이브 값을 100°C 32분으로 외삽하면 Hird 단위(351·81 bp)로 약 2.0 cycle.

    8.59 · (32/15) · k(100)/k(121) = 8.59 · 2.133 · 0.1073 ≈ 1.97.
    """
    extrap = 8.59 * (32 / 15) * rate_ratio(121.0, 100.0, 130.0)
    assert extrap == pytest.approx(2.0, rel=0.05)


def test_ddi_increase_identity():
    """같은 온도면 비가 1이라 ΔDDI = 482·k·t/ln2."""
    import numpy as np
    assert lit.ddi_increase(1e-5, 80.0, 130.0) == pytest.approx(482 * 1e-5 * 60 / np.log(2))
