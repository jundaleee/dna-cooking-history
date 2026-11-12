"""t_eq 단위 검출 한계와 해상도 검증."""
import numpy as np
import pytest

from rne.models import (
    ddi_from_lambda,
    ddi_per_teq,
    lod_60,
    log_teq_sd_approx,
    min_detectable_teq,
    teq_resolution_min,
)


def test_ddi_per_teq_consistent_with_ddi_from_lambda():
    """b·t_eq = ddi_from_lambda(k(80)·t_eq)."""
    k80 = 1.8e-5
    assert ddi_per_teq(k80) * 60.0 == pytest.approx(ddi_from_lambda(k80 * 60.0))
    assert ddi_per_teq(k80, 0.9) == pytest.approx(482 * k80 / np.log(1.9))


def test_min_detectable_teq_is_lod_over_slope():
    b = 0.766 / 60.0  # Brisco 수준 (80°C 60분에 0.766 cycle)
    assert min_detectable_teq(0.42, b) == pytest.approx(lod_60(0.42) / b)
    # σ 0.42면 LoD 0.635 cycle → 약 49.7 80°C-분
    assert min_detectable_teq(0.42, b) == pytest.approx(49.7, rel=0.01)


def test_min_detectable_teq_scaling_and_limits():
    """S2: σ에 비례, b에 반비례, 반복 4배면 절반, b = 0이면 무한대."""
    b = 0.01
    assert min_detectable_teq(0.84, b) == pytest.approx(2 * min_detectable_teq(0.42, b))
    assert min_detectable_teq(0.42, 2 * b) == pytest.approx(0.5 * min_detectable_teq(0.42, b))
    assert min_detectable_teq(0.42, b, n_rep=24) == pytest.approx(0.5 * min_detectable_teq(0.42, b))
    assert min_detectable_teq(0.42, 0.0) == float("inf")
    assert min_detectable_teq(0.0, b) == 0.0


def test_teq_resolution():
    b = 0.766 / 60.0
    assert teq_resolution_min(0.42, b) == pytest.approx(0.42 / b)
    assert teq_resolution_min(0.42, b, n_rep=4) == pytest.approx(0.21 / b)
    assert teq_resolution_min(0.42, 0.0) == float("inf")


def test_log_teq_sd_approx_matches_monte_carlo_when_teq_large():
    """t_eq ≫ t_res이면 델타 근사 t_res/t_eq가 몬테카를로 log 표준편차와 맞는다."""
    rng = np.random.default_rng(0)
    t_eq, t_res = 60.0, 3.0
    est = t_eq + t_res * rng.standard_normal(200_000)
    assert np.std(np.log(est)) == pytest.approx(log_teq_sd_approx(t_eq, t_res), rel=0.02)
    assert log_teq_sd_approx(0.0, 1.0) == float("inf")
