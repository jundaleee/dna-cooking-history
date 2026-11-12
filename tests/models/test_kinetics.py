"""rne.models.kinetics 검증."""
import numpy as np
import pytest

from rne.common import schema
from rne.models import k_chem, lam_A, lam_A_residual, lam_B, lam_B_intercept, rate_ratio


@pytest.mark.parametrize("Ea, expected", [(110, 32.4), (121, 45.9), (130, 61.0), (150, 114.8)])
def test_rate_ratio_matches_doc(Ea, expected):
    """01 문서 5-3 기준값의 근거: 50→80°C 속도 비 (상대 2%)."""
    assert rate_ratio(50, 80, Ea) == pytest.approx(expected, rel=0.02)


def test_h1_threshold_is_lower_ea_ratio():
    """schema.H1_RATIO_THRESHOLD(32)는 Ea 하한 110의 비와 같다."""
    assert rate_ratio(50, 80, schema.EA_RANGE_KJ[0]) == pytest.approx(schema.H1_RATIO_THRESHOLD, rel=0.02)


def test_k_chem_reference_and_vectorized():
    assert k_chem(80.0, 2e-5, 130.0) == pytest.approx(2e-5)
    k = k_chem(np.array(schema.TEMPERATURES_C), 1.0, 130.0)
    assert isinstance(k, np.ndarray) and np.all(np.diff(k) > 0)
    assert isinstance(k_chem(50.0, 1.0, 130.0), float)


def test_lam_A_linear_in_time():
    lam = lam_A(70.0, np.array([0.0, 15.0, 30.0, 60.0]), 1e-4, 2e-5, 130.0)
    slope = np.diff(lam) / np.diff([0.0, 15.0, 30.0, 60.0])
    assert lam[0] == pytest.approx(1e-4)
    assert np.allclose(slope, k_chem(70.0, 2e-5, 130.0))


@pytest.mark.parametrize("tau", [2.0, 5.0, 10.0])
def test_A_residual_intercept_over_slope_is_tau_for_all_temperatures(tau):
    """A': (절편 − λ₀)/기울기 = τ, 온도와 무관."""
    lam0 = 1e-4
    t = np.array(schema.TIMES_MIN)
    for T in schema.TEMPERATURES_C:
        lam = lam_A_residual(T, t, lam0, 2e-5, 130.0, tau)
        slope, intercept = np.polyfit(t, lam, 1)
        assert (intercept - lam0) / slope == pytest.approx(tau, rel=1e-9)


def test_A_residual_tau0_equals_A():
    t = np.array(schema.TIMES_MIN)
    assert np.allclose(lam_A_residual(60, t, 1e-4, 2e-5, 130, 0.0), lam_A(60, t, 1e-4, 2e-5, 130))


def test_B_to_B_intercept_large_q():
    """q가 클 때 t ≥ 15에서 B와 B'의 차이 < 1e-6."""
    t = np.array([15.0, 30.0, 60.0])
    for T in schema.TEMPERATURES_C:
        b = lam_B(T, t, 1e-4, 2e-5, 130.0, X=1e-3, q_per_min=5.0)
        bp = lam_B_intercept(T, t, 1e-4, 2e-5, 130.0, X=1e-3)
        assert np.max(np.abs(b - bp)) < 1e-6


def test_B_accepts_temperature_functions():
    X = lambda T: 1e-3 * np.exp(-(np.asarray(T) - 50.0) / 15.0)  # noqa: E731
    q = lambda T: 0.5 + 0 * np.asarray(T)  # noqa: E731
    b = lam_B(np.array([50.0, 80.0]), 60.0, 0.0, 0.0, 130.0, X, q)
    assert b[0] > b[1]
    assert b[0] == pytest.approx(1e-3 * (1 - np.exp(-30.0)))


def test_B_q0_or_X0_equals_A():
    t = np.array(schema.TIMES_MIN)
    a = lam_A(75, t, 1e-4, 2e-5, 130)
    assert np.allclose(lam_B(75, t, 1e-4, 2e-5, 130, X=0.0, q_per_min=1.0), a)
    assert np.allclose(lam_B(75, t, 1e-4, 2e-5, 130, X=1e-3, q_per_min=0.0), a)
