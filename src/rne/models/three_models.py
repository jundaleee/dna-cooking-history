"""세 가설의 DDI 모델: M1 아레니우스(산 촉매 탈퓨린화), M2 국소 융해, M3 효소 소진.

브로드캐스팅으로 계산한다(합성 실험에서 수백만 번 호출됨).

단위: 온도 °C(안에서 K로), 시간 분(t_e, 등가 가열 시간), k_An은 s⁻¹, λ는 nt⁻¹,
에너지 kJ/mol, DDI는 cycle.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from rne.common.schema import DELTA_BP, R_GAS

ArrayLike = Union[float, np.ndarray]
_K = 273.15
_LN10 = np.log(10.0)

#: An 등 2014 식의 계수 (log10 k = A − B·pH − C/T, k는 s⁻¹, T는 K, pH ≥ 2.5)
AN_A: float = 16.5
AN_B_PH: float = 0.982
AN_C_K: float = 5850.0
#: An 식의 온도 계수를 활성화 에너지로 환산한 값 C·ln10·R (≈ 112.0 kJ/mol)
EA_AN_KJ: float = AN_C_K * _LN10 * R_GAS / 1e3


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _out(x: np.ndarray) -> ArrayLike:
    return float(x) if np.ndim(x) == 0 else x


#: ea_kJ를 바꿀 때 속도를 고정하는 기준 온도 (실험 설계 50–80°C의 가운데, 가정)
T_ANCHOR_C: float = 65.0


def k_an(temperature_C: ArrayLike, pH: ArrayLike, ea_kJ: ArrayLike | None = None,
         T_anchor_C: float = T_ANCHOR_C) -> ArrayLike:
    """산 촉매 탈퓨린화 속도 k_An (s⁻¹, 퓨린 하나당).

    log10 k = 16.5 − 0.982·pH − 5850/T. 근거: 16번 문서 모델 1, An 등 2014
    (단일가닥 올리고, 50 mM 인산 완충액, pH ≥ 2.5).

    ea_kJ를 주면 온도 계수만 Ea로 바꾼다: k(T) = k_An(T_anchor)·exp[−Ea/R·(1/T − 1/T_anchor)].
    T_anchor에서는 An 식과 같은 값이다. 생성기에서 Ea 110–150 민감도를 보려고 둔 것이며
    고정 기준 온도는 가정이다. None이면 An 식 그대로(Ea 112.0).
    """
    T_K = _arr(temperature_C) + _K
    if ea_kJ is None:
        return _out(10.0 ** (AN_A - AN_B_PH * _arr(pH) - AN_C_K / T_K))
    Ta_K = T_anchor_C + _K
    k_anchor = 10.0 ** (AN_A - AN_B_PH * _arr(pH) - AN_C_K / Ta_K)
    return _out(k_anchor * np.exp(-(_arr(ea_kJ) * 1e3 / R_GAS) * (1.0 / T_K - 1.0 / Ta_K)))


def _ddi_from_dlam(dlam: np.ndarray, efficiency: float, ddi0: float) -> np.ndarray:
    """λ − λ₀ (nt⁻¹)를 DDI로: DDI₀ + 482·Δλ/ln(1+E)."""
    return ddi0 + DELTA_BP * dlam / np.log1p(efficiency)


def _chem_dlam(temperature_C, time_min, pH, f_pur, beta, ea_kJ=None) -> np.ndarray:
    """M1의 화학 손상 Δλ = (f_pur/β)·k_An·t_e (nt⁻¹). t_e는 분이라 60을 곱한다."""
    return (_arr(f_pur) / _arr(beta)) * _arr(k_an(temperature_C, pH, ea_kJ)) * (_arr(time_min) * 60.0)


def ddi_m1(temperature_C: ArrayLike, time_min: ArrayLike, pH: ArrayLike, f_pur: ArrayLike = 0.5,
           beta: ArrayLike = 2.4, efficiency: float = 1.0, ddi0: float = 0.0,
           ea_kJ: ArrayLike | None = None) -> ArrayLike:
    """M1 아레니우스 DDI (cycle).

    DDI⁽¹⁾ = DDI₀ + 482·(f_pur/β)·k_An(T, pH)·t_e / ln(1+E). 근거: 16번 문서 모델 1.

    Args:
        temperature_C: 가열 온도 (°C).
        time_min: 등가 가열 시간 t_e (분).
        pH: 시료 pH.
        f_pur: 표적의 퓨린 비율 (0–1).
        beta: 단일가닥 대 이중가닥 속도 비 (≥ 1).
        efficiency: 증폭 효율 E.
        ddi0: 가열 전 DDI (cycle).
        ea_kJ: None이면 An 식(Ea 112.0). 주면 k_an의 설명대로 온도 계수만 바꾼다 (kJ/mol).
    """
    return _out(_ddi_from_dlam(_chem_dlam(temperature_C, time_min, pH, f_pur, beta, ea_kJ), efficiency, ddi0))


def theta_melt(temperature_C: ArrayLike, T_m_C: ArrayLike, dH_kJ: ArrayLike) -> ArrayLike:
    """표적이 단일가닥인 비율 θ(T) (0–1).

    θ = 1/(1 + exp[ΔH/R·(1/T − 1/T_m)]). T = T_m이면 0.5. 근거: 16번 문서 모델 2.
    """
    T_K = _arr(temperature_C) + _K
    Tm_K = _arr(T_m_C) + _K
    z = (_arr(dH_kJ) * 1e3 / R_GAS) * (1.0 / T_K - 1.0 / Tm_K)
    # exp 넘침 방지: 1/(1+e^z) = expit(−z)
    return _out(0.5 * (1.0 - np.tanh(0.5 * z)))


def ddi_m2(temperature_C: ArrayLike, time_min: ArrayLike, pH: ArrayLike, T_m_C: ArrayLike,
           dH_kJ: ArrayLike, f_pur: ArrayLike = 0.5, beta: ArrayLike = 2.4,
           efficiency: float = 1.0, ddi0: float = 0.0, ea_kJ: ArrayLike | None = None) -> ArrayLike:
    """M2 국소 융해 DDI (cycle).

    k_eff = (f_pur/β)·k_An·[1 + (β − 1)·θ(T)], DDI⁽²⁾ = DDI₀ + 482·k_eff·t_e/ln(1+E).
    θ = 0이면 M1, θ = 1이면 단일가닥 속도 f_pur·k_An. 근거: 16번 문서 모델 2.
    ea_kJ는 ddi_m1과 같다.
    """
    boost = 1.0 + (_arr(beta) - 1.0) * _arr(theta_melt(temperature_C, T_m_C, dH_kJ))
    dlam = _chem_dlam(temperature_C, time_min, pH, f_pur, beta, ea_kJ) * boost
    return _out(_ddi_from_dlam(dlam, efficiency, ddi0))


def mu_enzyme(temperature_C: ArrayLike, time_min: ArrayLike, k_e_ref_per_min: ArrayLike,
              Ea_e_kJ: ArrayLike, k_d_ref_per_min: ArrayLike, Ea_d_kJ: ArrayLike,
              T_ref_C: float = 60.0) -> ArrayLike:
    """M3 효소 절단 누적량 μ_e(T, t) = X(T)·[1 − e^(−q(T)·t_e)] (nt⁻¹).

    X = k_e/k_d, q = k_d. k_e(효소 절단, nt⁻¹·min⁻¹)와 k_d(효소 비활성화, min⁻¹)는
    각각 아레니우스 k(T) = k_ref·exp[−Ea/R·(1/T − 1/T_ref)]로 둔다. 이 온도 형태는
    16번 문서에 없어 이 구현의 가정이다.
    """
    T_K = _arr(temperature_C) + _K
    inv = 1.0 / T_K - 1.0 / (T_ref_C + _K)
    k_e = _arr(k_e_ref_per_min) * np.exp(-(_arr(Ea_e_kJ) * 1e3 / R_GAS) * inv)
    k_d = _arr(k_d_ref_per_min) * np.exp(-(_arr(Ea_d_kJ) * 1e3 / R_GAS) * inv)
    t = _arr(time_min)
    # X·(1 − e^(−k_d t)) = k_e·t·(1 − e^(−x))/x, x = k_d t. k_d → 0에서도 안정
    x = k_d * t
    frac = np.where(x > 1e-8, -np.expm1(-x) / np.where(x > 1e-8, x, 1.0), 1.0 - 0.5 * x)
    return _out(k_e * t * frac)


def k_e_for_ratio(ratio_c: ArrayLike, pH: ArrayLike, rho: ArrayLike, Ea_e_kJ: ArrayLike,
                  k_d_ref_per_min: ArrayLike, Ea_d_kJ: ArrayLike, f_pur: ArrayLike = 0.5,
                  beta: ArrayLike = 2.4, ea_kJ: ArrayLike | None = None, T_ref_C: float = 60.0,
                  T_cal_C: float = 50.0, t_cal_min: float = 60.0) -> ArrayLike:
    """효소 크기를 화학 손상에 대한 비 c로 정했을 때의 k_e(T_ref) (nt⁻¹·min⁻¹).

    T_cal(기본 50°C), t_cal(기본 60분)에서 ρ·μ_e = c × 화학 Δλ(M1)가 되도록 푼다. μ_e는 k_e에
    정비례하므로 k_e = c·Δλ_chem / (ρ·μ_e|k_e=1)이다.
    ("50°C에서 효소 손상이 화학 손상의 0.5–5배", 07 문서 5-4의 가정). 생성기마다 같은 환산을 쓰게 하려고 둔다.
    """
    chem = _chem_dlam(T_cal_C, t_cal_min, pH, f_pur, beta, ea_kJ)
    unit = _arr(mu_enzyme(T_cal_C, t_cal_min, 1.0, Ea_e_kJ, k_d_ref_per_min, Ea_d_kJ, T_ref_C))
    return _out(_arr(ratio_c) * chem / (_arr(rho) * unit))


def ddi_m3(temperature_C: ArrayLike, time_min: ArrayLike, pH: ArrayLike, rho: ArrayLike,
           k_e_ref_per_min: ArrayLike, Ea_e_kJ: ArrayLike, k_d_ref_per_min: ArrayLike,
           Ea_d_kJ: ArrayLike, f_pur: ArrayLike = 0.5, beta: ArrayLike = 2.4,
           efficiency: float = 1.0, ddi0: float = 0.0, T_ref_C: float = 60.0,
           ea_kJ: ArrayLike | None = None, T_m_C: ArrayLike | None = None,
           dH_kJ: ArrayLike | None = None) -> ArrayLike:
    """M3 효소 소진 DDI (cycle): 화학 손상 + 효소 손상.

    DDI⁽³⁾ = DDI₀ + 482·[화학 Δλ + ρ·μ_e(T, t)]/ln(1+E).
    - 화학 항: T_m_C와 dH_kJ를 주면 M2(국소 융해, (f_pur/β)·k_An·[1 + (β − 1)·θ]·t_e),
      주지 않으면 M1((f_pur/β)·k_An·t_e). M2는 θ → 0에서 M1이 되므로 M1을 포함한다.
      M2 바탕은 "고온 융해 + 저온 효소" 세계를 표현하려고 둔 것이다
     . 16번 문서의 M3 정의는 M1 바탕이다
    - ρ는 mtDNA가 효소에 노출되는 정도 (0–1). ρ = 0이면 화학 항만 남는다
    """
    if (T_m_C is None) != (dH_kJ is None):
        raise ValueError("T_m_C와 dH_kJ는 함께 주거나 함께 비워야 한다")
    chem = _chem_dlam(temperature_C, time_min, pH, f_pur, beta, ea_kJ)
    if T_m_C is not None:
        chem = chem * (1.0 + (_arr(beta) - 1.0) * _arr(theta_melt(temperature_C, T_m_C, dH_kJ)))
    dlam = chem + _arr(rho) * _arr(
        mu_enzyme(temperature_C, time_min, k_e_ref_per_min, Ea_e_kJ, k_d_ref_per_min, Ea_d_kJ, T_ref_C))
    return _out(_ddi_from_dlam(dlam, efficiency, ddi0))
