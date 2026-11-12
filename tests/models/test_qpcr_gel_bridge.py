"""rne.models의 qPCR·겔·다리·검출 한계 검증."""
import numpy as np
import pytest

from rne.common import schema
from rne.models import (
    ct_from_lambda,
    d_statistic,
    ddi_from_lambda,
    ddi_johnston,
    gel_mass_density,
    johnston_intact_fraction,
    lambda_from_ddi,
    lod_60,
    s_tt,
)


# --- qPCR (4-2) ---
def test_ct_from_lambda_matches_formula():
    ct = ct_from_lambda(1e-3, 586, 22.0, template_ratio=0.5, efficiency=0.9)
    assert ct == pytest.approx(22.0 + (1e-3 * 586 - np.log(0.5)) / np.log(1.9))


def test_ddi_is_difference_of_cts_when_efficiencies_equal():
    lam, E = 7e-4, 0.95
    ddi = (ct_from_lambda(lam, schema.L586_BP, 23.0, 0.3, E)
           - ct_from_lambda(lam, schema.S104_BP, 21.0, 0.3, E))
    assert ddi == pytest.approx(ddi_from_lambda(lam, ddi0=2.0, efficiency=E))


@pytest.mark.parametrize("x", [0.0, 1e-5, 3e-4, 2e-3])
@pytest.mark.parametrize("E", [1.0, 0.9])
def test_lambda_ddi_roundtrip(x, E):
    assert lambda_from_ddi(ddi_from_lambda(x, 0.4, E), 0.4, E) == pytest.approx(x, abs=1e-15)


def test_ddi_from_lambda_vectorized():
    out = ddi_from_lambda(np.array([0.0, np.log(2) / 482]))
    assert np.allclose(out, [0.0, 1.0])


# --- 겔 (4-3) ---
def test_gel_mass_density_peak_at_1_over_r():
    x = np.linspace(100, 100000, 200001)
    r = 1 / 20000
    c = gel_mass_density(x, r)
    assert x[np.argmax(c)] == pytest.approx(1 / r, rel=1e-3)
    assert c.max() == pytest.approx(1.0, rel=1e-6)


def test_gel_mass_density_ladder_bands():
    x = np.arange(100, 1500, 1.0)
    c0 = gel_mass_density(x, 1 / 20000)
    c1 = gel_mass_density(x, 1 / 20000, ladder_amp=0.5, spacing_bp=200.0)
    diff = c1 - c0
    assert diff[np.searchsorted(x, 200)] == pytest.approx(0.5, rel=1e-3)
    assert diff[np.searchsorted(x, 300)] < 0.01


# --- 다리 (4-4) ---
def test_johnston_exponential_check():
    """지수 분포(L₀ = 20 kb)에서 586 bp 온전 비율 ≈ 0.9712, e^(−586/20000)과 차이 < 0.001.

    무작위 절단의 질량 분포 C_f ∝ f·e^(−f/L₀)를 1 bp 간격, 20·L₀까지 놓고 식 (2)에 넣는다.
    연속 극한에서는 정확히 e^(−r/L₀)이고, 이산 합의 (+1)항 때문에 약간 크다.
    """
    f = np.arange(1, 20 * 20000 + 1, dtype=float)
    m = f * np.exp(-f / 20000)
    p = johnston_intact_fraction(f, m, 586)
    assert p == pytest.approx(0.9712, abs=1e-4)
    assert abs(p - np.exp(-586 / 20000)) < 1e-3


def test_johnston_ddi_matches_deagle_for_exponential():
    """지수 분포에서 DDI^J ≈ 482·r/ln2 (Deagle 식)."""
    L0 = 5000.0
    f = np.arange(1, 40 * int(L0) + 1, dtype=float)
    m = f * np.exp(-f / L0)
    assert ddi_johnston(f, m) == pytest.approx(482 / L0 / np.log(2), rel=0.01)


def test_johnston_edge_cases():
    f = np.array([100.0, 600.0])
    assert johnston_intact_fraction(f, np.array([1.0, 0.0]), 586) == 0.0
    assert johnston_intact_fraction(np.array([586.0]), np.array([2.0]), 586) == pytest.approx(1 / 586)
    with pytest.raises(ValueError):
        johnston_intact_fraction(f, np.array([-1.0, 1.0]), 104)
    with pytest.raises(ValueError):
        johnston_intact_fraction(f, np.array([1.0]), 104)


def test_d_statistic():
    assert d_statistic(0.5, 0.2) == pytest.approx(0.3)


# --- 검출 한계 (5-2) ---
def test_s_tt_design():
    assert s_tt() == pytest.approx(6300.0)


@pytest.mark.parametrize("sigma, expected", [(0.42, 0.64), (0.60, 0.91)])
def test_lod_60_matches_doc(sigma, expected):
    assert lod_60(sigma) == pytest.approx(expected, abs=0.01)
