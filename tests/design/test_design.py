"""실험 전 계산(손상 모형, 정보량, 오류율 하한, 검정력, 배치표) 시험."""
import numpy as np
import pytest

from rne.design.damage import damage_ddi, design, expand, theta, theta_from_profile
from rne.design.error_budget import error_budget
from rne.design.information import (binary_min_error, design_information, entropy_bits, exact_mi,
                                    fano_lower_bound, gaussian_bits)
from rne.design.layout import eta2, make_layout
from rne.design.power import detection_power


def test_theta_reference_and_order():
    assert theta(80, 60) == pytest.approx(60.0)
    assert theta(50, 60) < theta(70, 60) < theta(80, 60)


def test_damage_ddi_scaled_to_D60():
    assert damage_ddi(np.array([80.0]), np.array([60.0]), 0.9)[0] == pytest.approx(0.9)
    assert damage_ddi(np.array([50.0]), np.array([15.0]), 0.9)[0] < 0.1


def test_design_reps_patterns():
    T, t, n = design([50, 80], [15, 30, 60], [3, 2, 3])
    assert list(n) == [3, 2, 3, 3, 2, 3] and list(T[:3]) == [50, 50, 50]
    assert design([50, 80], [15, 30], 6)[2].sum() == 24
    Ts, ts, cond = expand(T, t, n)
    assert len(Ts) == 16 and cond.max() == 5
    with pytest.raises(ValueError):
        design([50, 80], [15, 30], [1, 2, 3])


def test_exact_mi_limits():
    assert exact_mi([0.0, 0.0], [1, 1], 0.4) == pytest.approx(0.0, abs=1e-3)
    assert exact_mi([0.0, 100.0], [1, 1], 0.4) == pytest.approx(1.0, abs=1e-3)
    assert entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)


def test_fano_and_binary():
    w = np.ones(8)
    assert fano_lower_bound(0.0, w) == pytest.approx((3 - 1) / np.log2(7))
    assert fano_lower_bound(3.0, w) == 0.0
    err, chance = binary_min_error([0.0, 50.0], [1, 1], [False, True], 0.4)
    assert err == pytest.approx(0.0, abs=1e-6) and chance == pytest.approx(0.5)
    err, _ = binary_min_error([0.0, 0.0], [1, 1], [False, True], 0.4)
    assert err == pytest.approx(0.5, abs=1e-3)


def test_design_information_reference_values():
    r = design_information([50, 60, 70, 75, 80], [15, 30, 60], [3, 2, 3], D60=0.9, sigma=0.42, pass_at=(75, 15))
    assert r["I_bits"] == pytest.approx(0.2077, abs=1e-3)
    assert r["fano_min_error"] == pytest.approx(0.703, abs=1e-3)
    assert r["binary_min_error"] == pytest.approx(0.364, abs=1e-3)
    assert r["pass_fraction"] == pytest.approx(0.525)


def test_gaussian_bits():
    assert gaussian_bits(0) == 0.0 and gaussian_bits(-0.5) == 0.0
    assert gaussian_bits(np.sqrt(0.75)) == pytest.approx(1.0)


def test_error_budget_batch_offset_can_inflate():
    b = error_budget([50, 60, 70, 75, 80], [15, 30, 60], [3, 2, 3], D60=0.9, sigma=0.42, noise_levels=(0.84,),
                     batch_temps=(60, 70), batch_offsets=(-0.27,), lags_min=(), eas_kJ=(), signals=())
    assert b.loc[b["오차 원천"] == "시료 잡음", "변화_bits"].iloc[0] < 0
    assert b.loc[b["오차 원천"] == "효율 차 × 추출 배치 차", "변화_bits"].iloc[0] > 0


def test_power_small_and_large_signal():
    args = ([50, 60, 70, 75, 80], [15, 30, 60], [3, 2, 3])
    small = detection_power(*args, D60=0.02, n_sim=20, n_perm=49, seed=1)
    large = detection_power(*args, D60=10.0, n_sim=20, n_perm=49, seed=1)
    assert small <= 0.2 and large >= 0.9


def test_layout_counts_pairs_and_balance():
    df, score = make_layout([50, 60, 70, 75, 80], [15, 30, 60], [3, 2, 3], seed=3, n_search=3000)
    assert len(df) == 96 and df.well.is_unique
    assert df.kind.value_counts().to_dict() == {"sample": 80, "ntc": 8, "noprimer": 4, "mixonly": 2, "water": 2}
    rows_per_sample = df[df.kind == "sample"].groupby("sample_id").well.apply(lambda w: set(w.str[0]))
    assert all(len(r) == 1 for r in rows_per_sample)   # 두 구간이 같은 줄
    assert score < 0.05 and eta2([1, 1, 2, 2], [0, 0, 1, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        make_layout([50, 60, 70, 75, 80], [15, 30, 60], 6, n_search=10)


def test_theta_from_profile_matches_constant_temperature():
    tt = np.linspace(0, 30, 61)
    assert theta_from_profile(tt, np.full(tt.size, 70.0)) == pytest.approx(float(theta(70, 30)), rel=1e-9)
    lag = np.minimum(70.0, 20 + 50 * tt / 3)          # 3분 동안 20 → 70 °C
    assert theta_from_profile(tt, lag) < float(theta(70, 30))
