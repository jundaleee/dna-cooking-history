"""정상 증폭 범위 점검 시험(가상 곡선): 정상 곡선의 오거부율은 문턱의 명목값 근처, 실측에서 본 이상은 대부분 잡는다."""
import numpy as np
import pandas as pd

from rne.qc.normal_range import (MahalanobisOOD, curve_features, fit_detectors, inject, load_kinds,
                                 simulate_normal, to_matrix)

T_GRID = np.linspace(60, 95, 141)


def test_simulated_normal_features_obey_physics():
    rows = simulate_normal(200, "amp", T_GRID, np.random.default_rng(0))
    g = np.array([r["g_max"] for r in rows])
    # 배율의 이론 상한은 2이고, 측정 잡음 때문에 일부가 조금 넘는다
    assert np.nanmedian(g) < 2.0 and np.nanpercentile(g, 99) < 2.7 and to_matrix(rows, "amp").shape == (200, 3)


def test_detectors_false_reject_and_injected():
    _, check = fit_detectors(T_GRID, n=800, seed=1)
    assert check["amp"]["false_reject_rate_test"] < 0.12
    assert min(check["amp"]["detect_rate_injected"].values()) > 0.8
    assert check["neg"]["detect_rate_injected"]["neg_rises"] > 0.9


def test_score_nan_is_outside():
    rng0, rng1 = np.random.default_rng(0), np.random.default_rng(1)
    d = MahalanobisOOD().fit(rng0.normal(size=(200, 2)), rng1.normal(size=(200, 2)))
    assert np.isinf(d.score(np.array([[np.nan, 0.0]]))[0])


def test_curve_features_flat_curve():
    f = curve_features(np.full(40, 1e4), np.full(40, 3e4), T_GRID, np.linspace(1, 0.5, T_GRID.size))
    assert abs(f["rise"]) < 1e-9 and f["rox_fold"] == 1.0 and np.isnan(f["g_max"])


def test_load_kinds_and_inject(tmp_path):
    p = tmp_path / "layout.csv"
    pd.DataFrame({"well": ["A1", "A2"], "kind": ["sample", "ntc"]}).to_csv(p, index=False)
    assert load_kinds(p) == {"A1": "sample", "A2": "ntc"}
    assert inject([{"g_max": 1.9}], "fast_growth")[0]["g_max"] == 2.4
