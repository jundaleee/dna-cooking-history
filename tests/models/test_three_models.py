"""세 모델 검증."""
import numpy as np
import pytest

from rne.common import schema
from rne.models import three_models as tm
from rne.models import ddi_from_lambda, rate_ratio

TEMPS = np.array(schema.TEMPERATURES_C)

# 16번 문서 모델 1 표: 60분 ΔDDI (cycle, E = 1, f_pur = 0.5, β = 2.4)
M1_TABLE = {
    7.4: [0.001, 0.002, 0.008, 0.014, 0.024],
    6.5: [0.005, 0.019, 0.061, 0.107, 0.186],
    5.8: [0.026, 0.092, 0.298, 0.523, 0.905],
}


@pytest.mark.parametrize("pH", list(M1_TABLE))
def test_m1_table(pH):
    """표 값은 소수 셋째 자리 반올림이다. 반올림 폭(±0.0005) 안에서 맞아야 한다."""
    got = tm.ddi_m1(TEMPS, 60.0, pH)
    assert np.allclose(got, M1_TABLE[pH], atol=5e-4 + 1e-9)


def test_m1_key_values():
    assert tm.ddi_m1(80.0, 60.0, 7.4) == pytest.approx(0.0243, rel=0.01)
    assert tm.ddi_m1(80.0, 60.0, 5.8) == pytest.approx(0.905, rel=0.005)


def test_m1_ratio_80_50_about_34_5_independent_of_pH():
    for pH in (5.8, 6.5, 7.4):
        r = tm.ddi_m1(80.0, 60.0, pH) / tm.ddi_m1(50.0, 60.0, pH)
        assert r == pytest.approx(34.5, rel=0.005)


def test_ea_an_equivalent():
    """An 식의 5850 K는 Ea ≈ 112 kJ/mol이고, 그 Ea의 아레니우스 비와 같다."""
    assert tm.EA_AN_KJ == pytest.approx(112.0, abs=0.1)
    r = tm.ddi_m1(80.0, 60.0, 7.0) / tm.ddi_m1(50.0, 60.0, 7.0)
    assert r == pytest.approx(rate_ratio(50.0, 80.0, tm.EA_AN_KJ), rel=1e-9)


def test_m1_time_linear_and_zero_at_t0():
    t = np.array([0.0, 15.0, 30.0, 60.0])
    d = tm.ddi_m1(70.0, t, 6.5, ddi0=1.3)
    assert d[0] == pytest.approx(1.3)
    assert np.allclose(np.diff(d) / np.diff(t), (d[-1] - d[0]) / 60.0)


def test_m1_matches_ddi_from_lambda():
    dlam = 0.5 / 2.4 * tm.k_an(75.0, 6.0) * 30 * 60
    assert tm.ddi_m1(75.0, 30.0, 6.0, efficiency=0.9) == pytest.approx(ddi_from_lambda(dlam, efficiency=0.9))


def test_k_an_formula():
    assert tm.k_an(80.0, 7.4) == pytest.approx(10 ** (16.5 - 0.982 * 7.4 - 5850 / 353.15))


# --- M2 ---
def test_theta_limits():
    assert tm.theta_melt(85.0, 85.0, 400.0) == pytest.approx(0.5)
    assert tm.theta_melt(20.0, 85.0, 400.0) < 1e-6
    assert tm.theta_melt(200.0, 85.0, 400.0) > 1 - 1e-6
    assert np.all(np.diff(tm.theta_melt(TEMPS, 85.0, 400.0)) > 0)
    # 넘침 없이 극단값
    assert tm.theta_melt(-50.0, 85.0, 5000.0) == 0.0


def test_m2_equals_m1_when_theta_zero():
    """θ = 0 (T_m이 아주 높음)이면 M2 = M1. β = 1이어도 M2 = M1."""
    m1 = tm.ddi_m1(TEMPS, 60.0, 6.5)
    assert np.allclose(tm.ddi_m2(TEMPS, 60.0, 6.5, T_m_C=500.0, dH_kJ=400.0), m1, rtol=1e-9)
    assert np.allclose(tm.ddi_m2(TEMPS, 60.0, 6.5, 85.0, 400.0, beta=1.0),
                       tm.ddi_m1(TEMPS, 60.0, 6.5, beta=1.0))


def test_m2_theta_one_is_single_strand_rate():
    """θ = 1이면 k_eff = f_pur·k_An (단일가닥 속도), M1의 β배."""
    m2 = tm.ddi_m2(80.0, 60.0, 6.5, T_m_C=-100.0, dH_kJ=400.0)
    assert m2 == pytest.approx(2.4 * tm.ddi_m1(80.0, 60.0, 6.5), rel=1e-9)


def test_m2_time_linear_and_above_m1():
    t = np.array([15.0, 30.0, 60.0])
    d = tm.ddi_m2(80.0, t, 6.5, 85.0, 400.0)
    assert np.allclose(d / t, d[0] / t[0])
    assert np.all(tm.ddi_m2(TEMPS, 60.0, 6.5, 85.0, 400.0) >= tm.ddi_m1(TEMPS, 60.0, 6.5))


# --- M3 ---
ENZ = dict(k_e_ref_per_min=2e-4, Ea_e_kJ=50.0, k_d_ref_per_min=np.log(2) / 5.0, Ea_d_kJ=300.0)


def test_m3_equals_m1_when_rho_zero():
    assert np.allclose(tm.ddi_m3(TEMPS, 60.0, 6.5, rho=0.0, **ENZ), tm.ddi_m1(TEMPS, 60.0, 6.5))


def test_mu_enzyme_limits():
    """t → 0이면 0, t → ∞이면 X = k_e/k_d, k_d → 0이면 k_e·t."""
    assert tm.mu_enzyme(60.0, 0.0, **ENZ) == 0.0
    X = ENZ["k_e_ref_per_min"] / ENZ["k_d_ref_per_min"]
    assert tm.mu_enzyme(60.0, 1e4, **ENZ) == pytest.approx(X, rel=1e-9)
    assert tm.mu_enzyme(60.0, 30.0, 2e-4, 50.0, 0.0, 300.0) == pytest.approx(2e-4 * 30.0)
    assert tm.mu_enzyme(60.0, 30.0, 2e-4, 50.0, 1e-12, 300.0) == pytest.approx(2e-4 * 30.0, rel=1e-6)


def test_m3_low_temperature_curvature_and_ratio():
    """M3: 저온에서 DDI가 시간에 오목(완만해짐)하고, s(80)/s(50)이 M1의 34.5보다 작다."""
    t = np.array([15.0, 30.0, 60.0])
    d50 = tm.ddi_m3(50.0, t, 6.5, rho=0.3, **ENZ)
    slopes = np.diff(d50) / np.diff(t)
    assert slopes[1] < slopes[0]
    s = lambda T: np.polyfit(t, tm.ddi_m3(T, t, 6.5, rho=0.3, **ENZ), 1)[0]  # noqa: E731
    assert s(80.0) / s(50.0) < 34.5


def test_m3_with_m2_background():
    """M3 바탕을 M2로: ρ = 0이면 M2와 같고, 효소 항은 바탕과 무관하게 더해지며, T_m을 안 주면 기존 M1 바탕."""
    t = np.array(schema.TIMES_MIN)
    T = TEMPS[:, None]
    m2 = tm.ddi_m2(T, t, 6.5, 86.9, 400.0)
    assert np.allclose(tm.ddi_m3(T, t, 6.5, rho=0.0, **ENZ, T_m_C=86.9, dH_kJ=400.0), m2, rtol=1e-12)
    enz_m1 = tm.ddi_m3(T, t, 6.5, rho=0.3, **ENZ) - tm.ddi_m1(T, t, 6.5)
    enz_m2 = tm.ddi_m3(T, t, 6.5, rho=0.3, **ENZ, T_m_C=86.9, dH_kJ=400.0) - m2
    assert np.allclose(enz_m1, enz_m2, rtol=1e-9)
    # T_m이 아주 높으면(θ → 0) M2 바탕 = M1 바탕
    assert np.allclose(tm.ddi_m3(T, t, 6.5, rho=0.3, **ENZ, T_m_C=500.0, dH_kJ=400.0),
                       tm.ddi_m3(T, t, 6.5, rho=0.3, **ENZ), rtol=1e-9)
    with pytest.raises(ValueError):
        tm.ddi_m3(80.0, 60.0, 6.5, rho=0.3, **ENZ, T_m_C=86.9)


def test_m2_background_offsets_enzyme_ratio():
    """효소는 s(80)/s(50)을 낮추고 융해는 높인다. M2 바탕이면 M1 바탕보다 비가 크다."""
    t = np.array(schema.TIMES_MIN)
    s = lambda T, **kw: np.polyfit(t, tm.ddi_m3(T, t, 6.5, rho=0.3, **ENZ, **kw), 1)[0]  # noqa: E731
    r_m1 = s(80.0) / s(50.0)
    r_m2 = s(80.0, T_m_C=84.0, dH_kJ=400.0) / s(50.0, T_m_C=84.0, dH_kJ=400.0)
    assert r_m1 < 34.5
    assert r_m2 > r_m1


@pytest.mark.parametrize("c", [0.5, 1.6, 5.0])
@pytest.mark.parametrize("pH", [5.8, 7.4])
def test_k_e_for_ratio(c, pH):
    """k_e_for_ratio로 정한 효소는 50°C 60분에서 정확히 c × 화학 손상이다 (priors 3절)."""
    kw = dict(Ea_e_kJ=50.0, k_d_ref_per_min=np.log(2) / 5.0, Ea_d_kJ=300.0)
    ke = tm.k_e_for_ratio(c, pH, 0.4, **kw)
    enz = tm.ddi_m3(50.0, 60.0, pH, 0.4, ke, **kw) - tm.ddi_m1(50.0, 60.0, pH)
    assert enz == pytest.approx(c * tm.ddi_m1(50.0, 60.0, pH), rel=1e-9)
    arr = tm.k_e_for_ratio(np.array([0.5, 5.0]), pH, 0.4, **kw)
    assert arr[1] / arr[0] == pytest.approx(10.0)


def test_ea_override():
    """ea_kJ: None이면 An 식 그대로, 112.0이면 거의 같음, 기준 온도에서 값 고정, 비는 rate_ratio."""
    assert tm.ddi_m1(TEMPS, 60.0, 6.5, ea_kJ=None) == pytest.approx(tm.ddi_m1(TEMPS, 60.0, 6.5))
    assert np.allclose(tm.ddi_m1(TEMPS, 60.0, 6.5, ea_kJ=tm.EA_AN_KJ), tm.ddi_m1(TEMPS, 60.0, 6.5), rtol=1e-9)
    assert tm.k_an(65.0, 6.5, ea_kJ=150.0) == pytest.approx(tm.k_an(65.0, 6.5))
    for ea in (110.0, 130.0, 150.0):
        r = tm.ddi_m1(80.0, 60.0, 6.0, ea_kJ=ea) / tm.ddi_m1(50.0, 60.0, 6.0, ea_kJ=ea)
        assert r == pytest.approx(rate_ratio(50.0, 80.0, ea), rel=1e-9)
    # M2·M3도 같은 경로로 받는다
    assert tm.ddi_m2(80.0, 60.0, 6.5, 500.0, 400.0, ea_kJ=130.0) == pytest.approx(tm.ddi_m1(80.0, 60.0, 6.5, ea_kJ=130.0))
    assert tm.ddi_m3(80.0, 60.0, 6.5, 0.0, **ENZ, ea_kJ=130.0) == pytest.approx(tm.ddi_m1(80.0, 60.0, 6.5, ea_kJ=130.0))
    # 배열 Ea
    out = tm.ddi_m1(TEMPS[None, :], 60.0, 6.5, ea_kJ=np.array([[110.0], [150.0]]))
    assert out.shape == (2, 5)


def test_broadcasting_over_parameter_draws():
    """모수 표본(행) × 조건(열) 브로드캐스팅."""
    rng = np.random.default_rng(0)
    pH = rng.uniform(5.8, 7.4, size=(1000, 1))
    T = np.repeat(TEMPS, 3)[None, :]
    t = np.tile(schema.TIMES_MIN, 5)[None, :]
    out = tm.ddi_m2(T, t, pH, rng.uniform(80, 95, (1000, 1)), 400.0)
    assert out.shape == (1000, 15)
    assert np.all(np.isfinite(out))
