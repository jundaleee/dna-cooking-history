"""t_eq_min 검증."""
import numpy as np
import pytest

from rne.common import schema
from rne.models import k_chem, lam_A, rate_ratio, t_eq_min


def test_reference_temperature_returns_time():
    t = np.array(schema.TIMES_MIN)
    for Ea in (110.0, 130.0, 150.0):
        assert np.allclose(t_eq_min(80.0, t, Ea), t)


def test_70C_60min_decreases_with_Ea():
    vals = [t_eq_min(70.0, 60.0, Ea) for Ea in (110.0, 130.0, 150.0)]
    assert vals[0] > vals[1] > vals[2] > 0


def test_matches_rate_ratio():
    """50°C 60분 = 60 / (50→80 속도 비). Ea 130이면 약 0.98분."""
    assert t_eq_min(50.0, 60.0) == pytest.approx(60.0 / rate_ratio(50, 80, 130.0))
    assert t_eq_min(50.0, 60.0) == pytest.approx(0.984, rel=0.01)


def test_limits():
    """S2: t = 0이면 0, Ea → 0이면 온도와 무관하게 t."""
    assert t_eq_min(60.0, 0.0) == 0.0
    assert t_eq_min(60.0, 30.0, Ea_kJ=1e-9) == pytest.approx(30.0)


def test_model_A_depends_only_on_t_eq():
    """모델 A에서 λ − λ₀ = k(80)·t_eq."""
    T = np.array(schema.TEMPERATURES_C)
    lam = lam_A(T, 30.0, 1e-4, 2e-5, 130.0)
    assert np.allclose(lam - 1e-4, 2e-5 * t_eq_min(T, 30.0))


def test_vectorized_and_ref_temperature():
    out = t_eq_min(np.array([60.0, 70.0]), np.array([15.0, 15.0]), T_ref_C=70.0)
    assert out[1] == pytest.approx(15.0)
    assert out[0] == pytest.approx(15.0 * k_chem(60.0, 1.0, 130.0, 70.0))
