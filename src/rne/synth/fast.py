"""빠른 합성 실험 생성기: 실험 한 번(시료 약 90개)의 Ct104·Ct586·DDI 표를 벡터 연산으로 만든다.

**합성 데이터다.** 판정 기준과 학습 가능성을 점검하는 데만 쓰며, 여기서 나온 성능은 연구 결과가 아니다.
학습 계산이 수십만 번 부르므로 실험 하나가 수 밀리초 안에 끝나도록 numpy만 쓴다.

측정 모형 (paper/report.md Ⅱ장):
    Ct_L = c0_L + plate_L + (Δλ·L − ln N)/ln(1+E_L) + batch·w_L + ε,   L = 104, 586
    DDI  = Ct_586 − Ct_104
- Δλ(T, t): 모델(효과 없음, M1, M2, M3, 모델 밖 법칙)이 정하는 가열 손상(nt⁻¹)
- N: 시료마다 다른 주형량(로그정규). 두 판의 E가 같으면 DDI에서 지워지고, 다르면 일부 남는다
- plate_L: 판마다 한 값(표적별 판 분리이므로 두 표적에 다른 값)
- batch: 추출 배치마다 한 값. 억제 물질은 긴 표적을 더 늦춘다(w_586 > w_104)
- ε: 웰 Ct 잡음. 시료 간 생물학적 변동은 Δλ에 곱하지 않고 DDI 쪽 잡음(sigma_sample)으로 넣는다
- 결정 2: 40 cycle까지 증폭이 없을 때만 중도절단(ct ≥ max_cycles → 미검출). 실패 웰은 따로 결측
식은 `rne.models.three_models`를 쓴다(최종 계산도 이것으로 했다). 그 모듈이 없을 때만 이 파일 안의 같은 식(대체 구현)을 쓴다.
설계 기본값: 2차 판 배치(원액 세트 가열 90 + 양념 6).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from rne.common.schema import DELTA_BP, L586_BP, MAX_CYCLES, R_GAS, S104_BP, TEMPERATURES_C, TIMES_MIN

MODELS = ("none", "M1", "M2", "M1_enzyme", "M2_enzyme", "M3", "M23",
          "autocatalytic", "saturating", "temp_inversion", "threshold_step")
#: 네 세계(M1, M2, M1+효소, M2+효소). "M3"는 지시 문구대로 M2+효소의 별칭이다.
#: 다른 구현은 M3 = M1+효소, M23 = M2+효소로 쓴다 → "M23"도 M2+효소 별칭으로 받는다("M3"의 뜻은 사람이 정함)
WORLDS = ("M1", "M2", "M1_enzyme", "M2_enzyme")
IN_MODEL = ("none", "M1", "M2", "M1_enzyme", "M2_enzyme", "M3", "M23")
OUT_OF_MODEL = ("autocatalytic", "saturating", "temp_inversion", "threshold_step")
_K = 273.15
_E_UNIT = np.e - 1.0  # ln(1+E) = 1이 되는 E: 모델 함수 값 = 482·Δλ


# --------------------------------------------------------------------------- 모델 식


def _fallback_models():
    """rne.models.three_models가 없을 때 쓰는 대체 구현(같은 식, 같은 시그니처)."""

    def k_an(temperature_C, pH, ea_kJ=None, T_anchor_C=65.0):
        T = np.asarray(temperature_C, float)
        if ea_kJ is None:
            return 10.0 ** (16.5 - 0.982 * np.asarray(pH, float) - 5850.0 / (T + _K))
        k_anchor = 10.0 ** (16.5 - 0.982 * np.asarray(pH, float) - 5850.0 / (T_anchor_C + _K))
        return k_anchor * np.exp(-(np.asarray(ea_kJ, float) * 1e3 / R_GAS) * (1.0 / (T + _K) - 1.0 / (T_anchor_C + _K)))

    def _chem(T, t, pH, f_pur, beta, ea_kJ=None):
        return (np.asarray(f_pur, float) / np.asarray(beta, float)) * k_an(T, pH, ea_kJ) * (np.asarray(t, float) * 60.0)

    def _ddi(dlam, efficiency, ddi0):
        return ddi0 + DELTA_BP * dlam / np.log1p(efficiency)

    def ddi_m1(temperature_C, time_min, pH, f_pur=0.5, beta=2.4, efficiency=1.0, ddi0=0.0, ea_kJ=None):
        return _ddi(_chem(temperature_C, time_min, pH, f_pur, beta, ea_kJ), efficiency, ddi0)

    def theta_melt(temperature_C, T_m_C, dH_kJ):
        z = (np.asarray(dH_kJ, float) * 1e3 / R_GAS) * (1.0 / (np.asarray(temperature_C, float) + _K)
                                                        - 1.0 / (np.asarray(T_m_C, float) + _K))
        return 0.5 * (1.0 - np.tanh(0.5 * z))

    def ddi_m2(temperature_C, time_min, pH, T_m_C, dH_kJ, f_pur=0.5, beta=2.4, efficiency=1.0, ddi0=0.0, ea_kJ=None):
        boost = 1.0 + (np.asarray(beta, float) - 1.0) * theta_melt(temperature_C, T_m_C, dH_kJ)
        return _ddi(_chem(temperature_C, time_min, pH, f_pur, beta, ea_kJ) * boost, efficiency, ddi0)

    def mu_enzyme(temperature_C, time_min, k_e_ref_per_min, Ea_e_kJ, k_d_ref_per_min, Ea_d_kJ, T_ref_C=60.0):
        inv = 1.0 / (np.asarray(temperature_C, float) + _K) - 1.0 / (T_ref_C + _K)
        k_e = np.asarray(k_e_ref_per_min, float) * np.exp(-(np.asarray(Ea_e_kJ, float) * 1e3 / R_GAS) * inv)
        k_d = np.asarray(k_d_ref_per_min, float) * np.exp(-(np.asarray(Ea_d_kJ, float) * 1e3 / R_GAS) * inv)
        x = k_d * np.asarray(time_min, float)
        safe = np.where(x > 1e-8, x, 1.0)
        frac = np.where(x > 1e-8, -np.expm1(-x) / safe, 1.0 - 0.5 * x)
        return k_e * np.asarray(time_min, float) * frac

    def ddi_m3(temperature_C, time_min, pH, rho, k_e_ref_per_min, Ea_e_kJ, k_d_ref_per_min, Ea_d_kJ,
               f_pur=0.5, beta=2.4, efficiency=1.0, ddi0=0.0, T_ref_C=60.0, ea_kJ=None):
        dlam = _chem(temperature_C, time_min, pH, f_pur, beta, ea_kJ) + np.asarray(rho, float) * mu_enzyme(
            temperature_C, time_min, k_e_ref_per_min, Ea_e_kJ, k_d_ref_per_min, Ea_d_kJ, T_ref_C)
        return _ddi(dlam, efficiency, ddi0)

    return {"k_an": k_an, "ddi_m1": ddi_m1, "ddi_m2": ddi_m2, "ddi_m3": ddi_m3, "source": "fallback"}


@lru_cache(maxsize=1)
def _accepts_ea() -> frozenset:
    """지금 모델 식(ddi_m1)이 받는 인자 이름."""
    import inspect

    return frozenset(inspect.signature(model_functions()["ddi_m1"]).parameters)


@lru_cache(maxsize=1)
def model_functions() -> dict:
    """모델 식 모음. `rne.models.three_models`가 있으면 그것(source='three_models')."""
    try:
        from rne.models import three_models as tm  # 늦은 import

        return {"k_an": tm.k_an, "ddi_m1": tm.ddi_m1, "ddi_m2": tm.ddi_m2, "ddi_m3": tm.ddi_m3,
                "source": "three_models"}
    except ImportError:
        return _fallback_models()


def default_params() -> dict:
    """모수 기본값. **시연값이다.** 고정값처럼 쓰지 않는다.

    - 화학: pH 6.5, f_pur 0.5, β 2.4 / M2: T_m 85 °C, ΔH 400 kJ/mol / 효소(M1_enzyme, M2_enzyme = M3): ρ 0.3, 효소 k·Ea
    - scale: Δλ 전체에 곱하는 효과 크기 배수(판별 가능 지도용, 1 = 문서 식 그대로)
    - 모델 밖: ac_tau_min(자가촉매 시간 척도), sat_cap(포화 상한 Δλ의 배수), step_T_C(계단 온도)
    - 측정: E_short·E_long, c0_short, ddi0(가열 전 DDI)
    - 잡음(cycle): sigma_ct(웰), sigma_sample(시료 간 DDI), sigma_plate(판), sigma_batch(배치),
      batch_w_short·batch_w_long(배치 효과가 표적에 실리는 비율), sigma_logN(주형량 ln 표준편차)
    - p_fail: 웰 실패 확률(결측), max_cycles: 중도절단 기준
    - ct_noise_df: None이면 정규, 숫자면 같은 SD의 t 분포(꼬리 두꺼움). ct_noise_slope: Ct가 10 cycle 늘 때
      웰 Ct 잡음 SD가 늘어나는 비율(0 = 등분산).
    """
    return {
        "pH": 6.5, "f_pur": 0.5, "beta": 2.4, "ea_kJ": None,
        "T_m_C": 85.0, "dH_kJ": 400.0,
        "rho": 0.3, "k_e_ref_per_min": 2e-5, "Ea_e_kJ": 60.0, "k_d_ref_per_min": 0.05, "Ea_d_kJ": 200.0,
        "scale": 1.0,
        "ac_tau_min": 30.0, "sat_cap": 0.5, "step_T_C": 72.0,
        "E_short": 0.95, "E_long": 0.95, "c0_short": 12.0, "ddi0": 3.0,
        "sigma_ct": 0.3, "sigma_sample": 0.3, "sigma_plate": 0.5, "sigma_batch": 0.3,
        "batch_w_short": 0.3, "batch_w_long": 1.0, "sigma_logN": 0.8,
        "p_fail": 0.0, "max_cycles": float(MAX_CYCLES),
        "ct_noise_df": None, "ct_noise_slope": 0.0,
    }


def delta_lambda(model: str, temperature_C, time_min, p: dict) -> np.ndarray:
    """모델별 가열 손상 Δλ (nt⁻¹). 모델 함수는 E = e − 1, DDI₀ = 0으로 불러 482·Δλ를 얻는다."""
    f = model_functions()
    T = np.asarray(temperature_C, float)
    t = np.asarray(time_min, float)
    chem = dict(pH=p["pH"], f_pur=p["f_pur"], beta=p["beta"], efficiency=_E_UNIT, ddi0=0.0)
    if p.get("ea_kJ") is not None:  # 화학 Ea를 모수로(None = An 식의 112.0, 기울기 비 34.5 고정)
        if "ea_kJ" not in _accepts_ea():
            raise RuntimeError("모델 식이 ea_kJ를 받지 않는다(옛 three_models). ea_kJ를 받는 판이 필요")
        chem["ea_kJ"] = p["ea_kJ"]
    if model == "none":
        d = np.zeros_like(T * t)
    elif model == "M1":
        d = np.asarray(f["ddi_m1"](T, t, **chem))
    elif model == "M2":
        d = np.asarray(f["ddi_m2"](T, t, T_m_C=p["T_m_C"], dH_kJ=p["dH_kJ"], **chem))
    elif model in ("M1_enzyme", "M2_enzyme", "M3", "M23"):
        # 효소 항은 Δλ에 더해진다. M1 바탕 M3에서 M1을 빼 효소 항만 얻고
        # 바탕 화학(M1 또는 M2)에 더한다 → three_models의 ddi_m3와 같은 값
        m1e = np.asarray(f["ddi_m3"](T, t, rho=p["rho"], k_e_ref_per_min=p["k_e_ref_per_min"], Ea_e_kJ=p["Ea_e_kJ"],
                                     k_d_ref_per_min=p["k_d_ref_per_min"], Ea_d_kJ=p["Ea_d_kJ"], **chem))
        if model == "M1_enzyme":
            d = m1e
        else:  # M2_enzyme, M3(지시 문구), M23
            enzyme = m1e - np.asarray(f["ddi_m1"](T, t, **chem))
            d = np.asarray(f["ddi_m2"](T, t, T_m_C=p["T_m_C"], dH_kJ=p["dH_kJ"], **chem)) + enzyme
    elif model in OUT_OF_MODEL:
        m1 = np.asarray(f["ddi_m1"](T, t, **chem))
        if model == "autocatalytic":  # 시간에 위로 굽음: 같은 60분 값, 모양 t²
            d = m1 * (t / 60.0) * (1.0 + t / p["ac_tau_min"]) / (1.0 + 60.0 / p["ac_tau_min"])
        elif model == "saturating":  # 시간에 포화: 상한 = 80°C 60분 M1 × sat_cap
            cap = p["sat_cap"] * float(np.asarray(f["ddi_m1"](80.0, 60.0, **chem)))
            d = cap * -np.expm1(-m1 / max(cap, 1e-12))
        elif model == "temp_inversion":  # 온도가 높을수록 느림: 50↔80 대칭으로 뒤집음
            d = np.asarray(f["ddi_m1"](130.0 - T, t, **chem))
        else:  # threshold_step: 아레니우스가 아니라 문턱 온도 위에서만 일정 속도
            rate = float(np.asarray(f["ddi_m1"](80.0, 1.0, **chem)))
            d = np.where(T >= p["step_T_C"], rate * t, 0.0)
    else:
        raise ValueError(f"모르는 모델: {model} (가능: {MODELS})")
    return p["scale"] * d / DELTA_BP


# --------------------------------------------------------------------------- 실험 설정


def stock_config() -> dict:
    """원액 세트: 2차 판 S104 + 3차 판 L586, 표적별 판 분리, 기술 반복 1, 대조군 없음.

    격자 5 온도 × 3 시간 × 6 반복 = 가열 90개 + 양념 6개.
    양념은 (간장, 고추장) × 60·70·80 °C 하나씩. 양념 시료의 가열 시간은 배치에 없다 → 30분으로 둔 가정.
    희석 세트(4·5차 판)는 30분 6번째 반복 자리를 NTC·표준에 쓰므로 가열 85개다(`dilution_config`).
    배치: 추출 배치 배정을 모른다. 기본은 온도마다 하나.
    """
    return {
        "name": "stock",
        "temperatures_C": list(TEMPERATURES_C), "times_min": list(TIMES_MIN), "n_rep": 6,
        "drop_rep_at_time": {},
        "marinade": [(80.0, 30.0, "soy"), (70.0, 30.0, "soy"), (60.0, 30.0, "soy"),
                     (80.0, 30.0, "gochujang"), (70.0, 30.0, "gochujang"), (60.0, 30.0, "gochujang")],
        "marinade_shift": {"soy": 0.0, "gochujang": 0.0},
        "batch_by": "temperature",  # 'temperature' | 'random' | 'single'
        "n_batches": 5,
        "n_tech_rep": 1,
        "ct_offset": 0.0,  # 두 표적에 똑같이 더하는 Ct 이동(수동). 희석은 아래 dilution으로
        "dilution": 1.0,  # 주형 희석 배수: ln N에서 ln(dilution)을 빼 표적마다 1/ln(1+E)로 들어간다
        "n_missing": 0,
    }


def dilution_config(dilution: float = 100.0) -> dict:
    """희석 세트: 원액 세트와 같은 시료, 주형을 dilution배 묽혀 Ct가 늦어진다.

    NTC 1 + 표준 4는 30분 6번째 반복 자리 — 시료 표에는 나오지 않는다.
    """
    c = stock_config()
    c.update(name=f"dilution_{dilution:g}", drop_rep_at_time={30.0: [6]}, dilution=float(dilution))
    return c


@lru_cache(maxsize=32)
def _design(key: tuple) -> dict:
    """설정의 고정된 부분(시료 목록)을 한 번만 만든다. key는 _design_key 결과."""
    temps, times, n_rep, drop, marinade, n_batches, batch_by = key
    drop = dict(drop)
    T, t, r, grp, mar = [], [], [], [], []
    for Ti in temps:
        for ti in times:
            for rep in range(1, n_rep + 1):
                if rep in drop.get(ti, ()):
                    continue
                T.append(Ti); t.append(ti); r.append(rep); grp.append("heat"); mar.append("none")
    count: dict = {}
    for Ti, ti, kind in marinade:
        count[(Ti, ti, kind)] = count.get((Ti, ti, kind), 0) + 1
        T.append(Ti); t.append(ti); r.append(count[(Ti, ti, kind)]); grp.append("marinade"); mar.append(kind)
    T = np.array(T, float); t = np.array(t, float); r = np.array(r, int)
    grp = np.array(grp); mar = np.array(mar)
    prefix = np.where(grp == "heat", "C", "M")
    sid = np.array([f"{p}-{int(a):03d}-{int(b)}-{c}" + ("" if m == "none" else f"-{m}")
                    for p, a, b, c, m in zip(prefix, T, t, r, mar)])
    if batch_by == "temperature":
        uniq = sorted(set(temps))
        batch_idx = np.array([uniq.index(x) % n_batches for x in T])
    elif batch_by == "single":
        batch_idx = np.zeros(len(T), int)
    else:
        batch_idx = None  # 'random': 실험마다 뽑는다
    return {"T": T, "t": t, "rep": r, "group": grp, "marinade": mar, "sample_id": sid, "batch_idx": batch_idx}


def _design_key(config: dict) -> tuple:
    drop = tuple(sorted((float(k), tuple(v)) for k, v in (config.get("drop_rep_at_time") or {}).items()))
    return (tuple(config["temperatures_C"]), tuple(config["times_min"]), int(config["n_rep"]), drop,
            tuple(tuple(m) for m in config.get("marinade", ())), int(config.get("n_batches", 1)),
            config.get("batch_by", "temperature"))


# --------------------------------------------------------------------------- 실험 한 번


def _ct_noise(rng, sigma, reps, ct_true, c_ref, p):
    """웰 Ct 잡음(기술 반복 평균). 정규 또는 t(ct_noise_df), Ct 수준에 따른 SD 증가(ct_noise_slope)."""
    n = len(ct_true)
    if sigma <= 0:
        return np.zeros(n)
    df = p.get("ct_noise_df")
    if df:
        z = rng.standard_t(df, (reps, n)) * np.sqrt((df - 2.0) / df) if df > 2 else rng.standard_t(df, (reps, n))
    else:
        z = rng.standard_normal((reps, n))
    scale = sigma * np.clip(1.0 + p.get("ct_noise_slope", 0.0) * (ct_true - c_ref) / 10.0, 0.1, None)
    return z.mean(0) * scale


def simulate_experiment(model: str, params: dict | None, config: dict | None, rng: np.random.Generator,
                        as_frame: bool = True):
    """합성 실험 한 번. 반환: DataFrame(as_frame) 또는 열 이름 → 배열 dict(빠른 경로).

    열: sample_id, group, marinade, temperature_C, time_min, replicate, extraction_batch,
    ct_short, ct_long, ddi, ddi_censored, ddi_lower, failed_short, failed_long, ddi_true
    - 중도절단: Ct ≥ max_cycles면 미검출. 586이 미검출이고 104가 검출이면 ddi_censored,
      ddi_lower = max_cycles − ct_short. 104가 미검출이거나 실패면 ddi 비움
    - 실패 웰(p_fail): 그 Ct는 결측(failed_*=True), 중도절단과 구별
    - n_missing: 시료를 무작위로 뺀다(표본 손실)
    - ddi_true: 잡음 없는 모델 DDI(점검용. 분석 코드는 쓰지 않는다)
    """
    p = default_params()
    if params:
        p.update(params)
    cfg = stock_config()
    if config:
        cfg.update(config)
    d = _design(_design_key(cfg))
    n = len(d["T"])
    T, t = d["T"], d["t"]
    dlam = delta_lambda(model, T, t, p)
    lnS, lnL = np.log1p(p["E_short"]), np.log1p(p["E_long"])
    # 시료 간 변동(DDI에 들어감): Δλ에 더해 긴 표적 쪽에 싣는다
    bio = rng.normal(0.0, p["sigma_sample"], n) if p["sigma_sample"] > 0 else np.zeros(n)
    lnN = rng.normal(0.0, p["sigma_logN"], n) if p["sigma_logN"] > 0 else np.zeros(n)
    ln_dil = float(np.log(float(cfg.get("dilution", 1.0))))  # 희석: 두 표적 모두 주형 ln N이 ln d만큼 준다
    n_b = max(1, int(cfg.get("n_batches", 1)))
    bidx = d["batch_idx"] if d["batch_idx"] is not None else rng.integers(0, n_b, n)
    beff = rng.normal(0.0, p["sigma_batch"], n_b)[bidx] if p["sigma_batch"] > 0 else np.zeros(n)
    plate_s, plate_l = (rng.normal(0.0, p["sigma_plate"], 2) if p["sigma_plate"] > 0 else (0.0, 0.0))
    shift = np.array([cfg.get("marinade_shift", {}).get(m, 0.0) for m in d["marinade"]]) if (d["group"] == "marinade").any() else 0.0
    off = float(cfg.get("ct_offset", 0.0))
    c0_s = p["c0_short"] + off
    c0_l = c0_s + p["ddi0"]
    ct_s_true = c0_s + plate_s + (dlam * S104_BP - lnN + ln_dil) / lnS + p["batch_w_short"] * beff
    ct_l_true = c0_l + plate_l + (dlam * L586_BP - lnN + ln_dil) / lnL + p["batch_w_long"] * beff + bio + shift
    reps = max(1, int(cfg.get("n_tech_rep", 1)))
    sct = p["sigma_ct"]
    ct_s = ct_s_true + _ct_noise(rng, sct, reps, ct_s_true, c0_s, p)
    ct_l = ct_l_true + _ct_noise(rng, sct, reps, ct_l_true, c0_s, p)
    mx = p["max_cycles"]
    fail_s = rng.random(n) < p["p_fail"] if p["p_fail"] > 0 else np.zeros(n, bool)
    fail_l = rng.random(n) < p["p_fail"] if p["p_fail"] > 0 else np.zeros(n, bool)
    und_s = ct_s >= mx
    und_l = ct_l >= mx
    ct_s = np.where(fail_s | und_s, np.nan, ct_s)
    ct_l = np.where(fail_l | und_l, np.nan, ct_l)
    has_s = ~np.isnan(ct_s)
    cens = has_s & und_l & ~fail_l
    ddi = np.where(has_s & ~np.isnan(ct_l), ct_l - ct_s, np.nan)
    lower = np.where(cens, mx - ct_s, np.nan)
    ddi_true = (c0_l - c0_s) + (dlam * L586_BP + ln_dil) / lnL - (dlam * S104_BP + ln_dil) / lnS + shift
    keep = np.ones(n, bool)
    k = int(cfg.get("n_missing", 0))
    if k > 0:
        keep[rng.choice(n, size=min(k, n), replace=False)] = False
    out = {
        "sample_id": d["sample_id"], "group": d["group"], "marinade": d["marinade"],
        "temperature_C": T, "time_min": t, "replicate": d["rep"],
        "extraction_batch": np.char.add("B", (bidx + 1).astype(str)),
        "ct_short": ct_s, "ct_long": ct_l, "ddi": ddi, "ddi_censored": cens, "ddi_lower": lower,
        "failed_short": fail_s, "failed_long": fail_l, "ddi_true": np.broadcast_to(ddi_true, (n,)).astype(float),
    }
    if k > 0:
        out = {key: np.asarray(v)[keep] for key, v in out.items()}
    if not as_frame:
        return out
    return pd.DataFrame(out)


def experiment_meta(model: str, params: dict | None, config: dict | None, seed: int) -> dict:
    """재현용 기록: 모델, 쓴 모수 전체, 설정 이름, 시드, 식의 출처(three_models / fallback)."""
    p = default_params()
    p.update(params or {})
    c = stock_config()
    c.update(config or {})
    return {"model": model, "params": p, "config": c.get("name"), "seed": seed,
            "model_source": model_functions()["source"],
            "warning": ("모델 식이 사본(fallback)이다 — rne.models.three_models와 갈라질 수 있다"
                        if model_functions()["source"] == "fallback" else "")}
