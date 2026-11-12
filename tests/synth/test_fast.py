"""synth/fast.py 시험. 합성(코드 점검용, 결과 아님)."""
import time

import numpy as np
import pytest

from rne.common.schema import DELTA_BP

from rne.synth.fast import (
    IN_MODEL,
    MODELS,
    OUT_OF_MODEL,
    _fallback_models,
    default_params,
    delta_lambda,
    dilution_config,
    model_functions,
    simulate_experiment,
    stock_config,
)

QUIET = dict(sigma_ct=0.0, sigma_sample=0.0, sigma_plate=0.0, sigma_batch=0.0, sigma_logN=0.0)


def test_stock_design_counts():
    d = simulate_experiment("M1", None, None, np.random.default_rng(0))
    assert len(d) == 96
    assert (d.group == "heat").sum() == 90 and (d.group == "marinade").sum() == 6
    heat = d[d.group == "heat"]
    assert heat.groupby("time_min").size().to_dict() == {15.0: 30, 30.0: 30, 60.0: 30}
    assert d["sample_id"].is_unique


@pytest.mark.parametrize("pH,expected", [(7.4, 0.024), (6.5, 0.186), (5.8, 0.905)])
def test_m1_table_docs16(pH, expected):
    # M1 표: 80 °C 60분 ΔDDI (E = 1, f_pur 0.5, β 2.4)
    f = model_functions()
    assert f["ddi_m1"](80.0, 60.0, pH, efficiency=1.0) == pytest.approx(expected, rel=0.03)


def test_fallback_matches_selected_source():
    fb = _fallback_models()
    f = model_functions()
    T = np.array([50.0, 70.0, 80.0]); t = np.array([15.0, 30.0, 60.0])
    assert np.allclose(fb["ddi_m1"](T, t, 6.0), f["ddi_m1"](T, t, 6.0))
    assert np.allclose(fb["ddi_m2"](T, t, 6.0, 85.0, 400.0), f["ddi_m2"](T, t, 6.0, 85.0, 400.0))
    a = fb["ddi_m3"](T, t, 6.0, 0.3, 2e-5, 60.0, 0.05, 200.0)
    b = f["ddi_m3"](T, t, 6.0, 0.3, 2e-5, 60.0, 0.05, 200.0)
    assert np.allclose(a, b)


def test_ratio_34_and_linearity_m1():
    p = default_params()
    s80 = delta_lambda("M1", 80.0, 60.0, p) - delta_lambda("M1", 80.0, 15.0, p)
    s50 = delta_lambda("M1", 50.0, 60.0, p) - delta_lambda("M1", 50.0, 15.0, p)
    assert s80 / s50 == pytest.approx(34.5, rel=0.03)
    d = delta_lambda("M1", 70.0, np.array([15.0, 30.0, 60.0]), p)
    assert d[1] / d[0] == pytest.approx(2.0) and d[2] / d[0] == pytest.approx(4.0)


def test_m2_m3_reduce_to_m1():
    p = default_params()
    T = np.array([50.0, 80.0]); t = np.array([30.0, 60.0])
    m1 = delta_lambda("M1", T, t, p)
    assert np.allclose(delta_lambda("M2", T, t, {**p, "T_m_C": 400.0}), m1)  # θ → 0
    assert np.allclose(delta_lambda("M1_enzyme", T, t, {**p, "rho": 0.0}), m1)
    m2 = delta_lambda("M2", T, t, p)
    assert np.allclose(delta_lambda("M2_enzyme", T, t, {**p, "rho": 0.0}), m2)  # ρ = 0이면 바탕 화학만
    assert np.allclose(delta_lambda("M3", T, t, p), delta_lambda("M2_enzyme", T, t, p))  # M3 = M2+효소(새 정의)
    assert np.all(m2 >= m1)
    assert np.all(delta_lambda("M1_enzyme", T, t, p) >= m1)
    # 효소 항은 바탕과 무관하게 같다
    assert np.allclose(delta_lambda("M2_enzyme", T, t, p) - m2, delta_lambda("M1_enzyme", T, t, p) - m1)


def test_m2_enzyme_matches_three_models():
    """three_models의 ddi_m3가 있으면 그것과 같은 값."""
    tm = pytest.importorskip("rne.models.three_models")
    import inspect

    if "T_m_C" not in inspect.signature(tm.ddi_m3).parameters:
        pytest.skip("이 브랜치의 three_models.ddi_m3는 M2 바탕 인자가 아직 없다")
    p = default_params()
    T = np.array([50.0, 60.0, 70.0, 80.0]); t = np.array([15.0, 30.0, 60.0, 60.0])
    ref = tm.ddi_m3(T, t, p["pH"], p["rho"], p["k_e_ref_per_min"], p["Ea_e_kJ"], p["k_d_ref_per_min"], p["Ea_d_kJ"],
                    f_pur=p["f_pur"], beta=p["beta"], efficiency=np.e - 1.0, T_m_C=p["T_m_C"], dH_kJ=p["dH_kJ"])
    assert np.allclose(delta_lambda("M2_enzyme", T, t, p) * DELTA_BP, ref)


def test_out_of_model_shapes():
    p = {**default_params(), "pH": 5.8}
    t = np.array([15.0, 30.0, 60.0])
    lin = delta_lambda("M1", 80.0, t, p)
    ac = delta_lambda("autocatalytic", 80.0, t, p)
    sat = delta_lambda("saturating", 80.0, t, p)
    assert ac[2] == pytest.approx(lin[2]) and ac[0] < lin[0]  # 위로 굽음
    assert (sat[2] - sat[1]) < (sat[1] - sat[0]) * 2  # 포화: 직선보다 덜 오름
    inv = delta_lambda("temp_inversion", np.array([50.0, 80.0]), 60.0, p)
    assert inv[0] > inv[1]
    step = delta_lambda("threshold_step", np.array([60.0, 75.0]), 60.0, p)
    assert step[0] == 0 and step[1] > 0
    with pytest.raises(ValueError):
        delta_lambda("nope", 50.0, 15.0, p)
    assert set(IN_MODEL) | set(OUT_OF_MODEL) == set(MODELS)


def test_noise_free_ddi_equals_truth_and_template_cancels():
    rng = np.random.default_rng(1)
    p = {**QUIET, "sigma_logN": 1.5, "pH": 5.8}
    d = simulate_experiment("M1", p, None, rng)
    assert np.allclose(d["ddi"], d["ddi_true"])  # E 같으면 주형량이 지워짐
    d2 = simulate_experiment("M1", {**p, "E_long": 0.80}, None, rng)
    assert not np.allclose(d2["ddi"], d2["ddi_true"])  # E가 다르면 ln N 항이 남음


def test_censoring_at_40_and_failures_and_missing():
    rng = np.random.default_rng(2)
    d = simulate_experiment("none", {**QUIET, "c0_short": 30.0, "ddi0": 12.0}, None, rng)
    assert d["ddi_censored"].all() and d["ddi"].isna().all()
    assert np.allclose(d["ddi_lower"], 10.0)
    f = simulate_experiment("none", {"p_fail": 0.5}, None, rng)
    assert f["failed_short"].mean() == pytest.approx(0.5, abs=0.15)
    assert f.loc[f.failed_long, "ddi"].isna().all() and not f.loc[f.failed_long, "ddi_censored"].any()
    m = simulate_experiment("none", None, {"n_missing": 7}, rng)
    assert len(m) == 89


def test_noise_levels():
    rng = np.random.default_rng(3)
    p = {**QUIET, "sigma_ct": 0.3}
    dd = np.concatenate([simulate_experiment("none", p, None, rng)["ddi"].to_numpy() for _ in range(50)])
    assert np.std(dd) == pytest.approx(0.3 * np.sqrt(2), rel=0.1)


def test_dilution_and_batches():
    rng = np.random.default_rng(4)
    a = simulate_experiment("none", QUIET, None, rng)
    b = simulate_experiment("none", QUIET, dilution_config(100.0), rng)
    assert (b["ct_short"] - a["ct_short"]).mean() == pytest.approx(np.log(100) / np.log(1.95), rel=0.01)
    c = simulate_experiment("none", None, {"batch_by": "random", "n_batches": 3}, rng)
    assert c["extraction_batch"].nunique() <= 3
    s = simulate_experiment("none", None, {"batch_by": "single"}, rng)
    assert s["extraction_batch"].nunique() == 1


def test_fast_enough():
    rng = np.random.default_rng(5)
    cfg = stock_config()
    simulate_experiment("M2", None, cfg, rng, as_frame=False)
    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        simulate_experiment("M3", None, cfg, rng, as_frame=False)
    per_ms = (time.perf_counter() - t0) / n * 1e3
    assert per_ms < 5.0, per_ms


def test_heavy_tail_and_heteroscedastic_noise():
    rng = np.random.default_rng(6)
    base = {**QUIET, "sigma_ct": 0.3}
    norm = np.concatenate([simulate_experiment("none", base, None, rng)["ddi"].to_numpy() for _ in range(80)])
    heavy = np.concatenate([simulate_experiment("none", {**base, "ct_noise_df": 4}, None, rng)["ddi"].to_numpy()
                            for _ in range(80)])
    assert np.std(heavy) == pytest.approx(np.std(norm), rel=0.15)  # 같은 SD
    k = lambda x: np.mean(((x - x.mean()) / x.std()) ** 4) - 3
    assert k(heavy) > k(norm) + 0.5
    hs = simulate_experiment("none", {**base, "ct_noise_slope": 1.0, "ddi0": 10.0}, None, rng)
    assert hs["ddi"].notna().all()


def test_dilution_design_counts():
    """희석 세트는 30분 6번째 반복 자리에 NTC·표준이 들어가 가열 85 + 양념 6이다."""
    d = simulate_experiment("M1", None, dilution_config(), np.random.default_rng(0))
    assert (d.group == "heat").sum() == 85 and (d.group == "marinade").sum() == 6
    assert d[d.group == "heat"].groupby("time_min").size().to_dict() == {15.0: 30, 30.0: 25, 60.0: 30}


def test_stock_marinade_matches_layout():
    """원액 세트 양념 = 2차 판 배치 H7–H12의 (간장, 고추장) × 60·70·80 °C."""
    m = simulate_experiment("none", None, None, np.random.default_rng(0))
    m = m[m.group == "marinade"]
    assert sorted(zip(m.marinade, m.temperature_C)) == sorted(
        (k, T) for k in ("soy", "gochujang") for T in (60.0, 70.0, 80.0))


def test_chemistry_ea_parameter_changes_slope_ratio():
    """ea_kJ를 주면 M1 기울기 비 s(80)/s(50)이 Ea의 아레니우스 비가 된다(None이면 34.5)."""
    from rne.common.schema import R_GAS as R

    p = default_params()
    for ea, expect in ((None, 34.5), (130.0, np.exp(130e3 / R * (1 / 323.15 - 1 / 353.15)))):
        d = delta_lambda("M1", np.array([50.0, 80.0]), np.array([60.0, 60.0]), {**p, "ea_kJ": ea})
        assert d[1] / d[0] == pytest.approx(expect, rel=2e-3)


def test_dilution_enters_through_template_term():
    """희석은 ln N 경로 → E가 다르면 DDI가 (1/ln(1+E_L) − 1/ln(1+E_S))·ln d만큼 바뀐다."""
    p = {**QUIET, "E_short": 0.95, "E_long": 0.85}
    a = simulate_experiment("none", p, None, np.random.default_rng(0))
    b = simulate_experiment("none", p, dilution_config(100.0), np.random.default_rng(0))
    expect = (1 / np.log1p(0.85) - 1 / np.log1p(0.95)) * np.log(100.0)
    assert expect == pytest.approx(0.590, abs=0.005)
    assert b["ddi"].mean() - a["ddi"].mean() == pytest.approx(expect, abs=1e-9)
    assert np.allclose(b["ddi"], b["ddi_true"])
    same = {**p, "E_long": 0.95}
    c = simulate_experiment("none", same, dilution_config(100.0), np.random.default_rng(0))
    assert c["ddi"].mean() == pytest.approx(simulate_experiment("none", same, None, np.random.default_rng(0))["ddi"].mean())


def test_fallback_matches_three_models_when_present():
    """사본 식과 rne.models.three_models가 같다(있을 때만). 사본이면 meta에 경고."""
    from rne.synth import fast

    fb = fast._fallback_models()
    tm = pytest.importorskip("rne.models.three_models")
    T = np.array([50.0, 65.0, 80.0]); t = np.array([15.0, 30.0, 60.0])
    assert np.allclose(fb["ddi_m1"](T, t, 6.5), tm.ddi_m1(T, t, 6.5))
    assert np.allclose(fb["ddi_m2"](T, t, 6.5, 85.0, 400.0), tm.ddi_m2(T, t, 6.5, 85.0, 400.0))


def test_meta_warns_on_fallback():
    from rne.synth.fast import experiment_meta, model_functions

    m = experiment_meta("M1", None, None, 0)
    assert (m["warning"] != "") == (model_functions()["source"] == "fallback")


def test_m23_alias_is_m2_enzyme():
    """다른 구현의 이름(M23 = M2+효소)도 받는다."""
    p = default_params()
    T = np.array([50.0, 80.0]); t = np.array([30.0, 60.0])
    assert np.allclose(delta_lambda("M23", T, t, p), delta_lambda("M2_enzyme", T, t, p))
