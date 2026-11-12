"""손상 동역학: 시간 모델 A, A', B, B'.

온도는 °C로 받아 안에서 K로 바꾼다. 활성화 에너지는 kJ/mol.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np

from rne.common.schema import R_GAS

ArrayLike = Union[float, np.ndarray]
# X, q는 스칼라 또는 "온도(°C) → 값" 함수 (00 문서 3장 lam_B 주석)
TempParam = Union[float, np.ndarray, Callable[[ArrayLike], ArrayLike]]

KELVIN_OFFSET: float = 273.15


def _to_kelvin(temperature_C: ArrayLike) -> ArrayLike:
    """°C를 K로 바꾼다."""
    return np.asarray(temperature_C, dtype=float) + KELVIN_OFFSET


def _resolve(param: TempParam, temperature_C: ArrayLike) -> ArrayLike:
    """스칼라면 그대로, 함수면 온도에서 계산한 값을 돌려준다."""
    if callable(param):
        return np.asarray(param(temperature_C), dtype=float)
    return np.asarray(param, dtype=float)


def _out(x: np.ndarray) -> ArrayLike:
    """0차원 배열은 float으로 돌려준다."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def k_chem(temperature_C: ArrayLike, k_ref_per_min: float, Ea_kJ: float,
           T_ref_C: float = 80.0) -> ArrayLike:
    """아레니우스 속도 k(T) [nt⁻¹·min⁻¹].

    k(T) = k_ref · exp(−Ea/R · (1/T − 1/T_ref)), T는 K.

    Args:
        temperature_C: 온도 (°C).
        k_ref_per_min: 기준 온도에서의 속도 (nt당 분당 손상).
        Ea_kJ: 활성화 에너지 (kJ/mol).
        T_ref_C: 기준 온도 (°C), 기본 80.
    """
    T_K = _to_kelvin(temperature_C)
    T_ref_K = T_ref_C + KELVIN_OFFSET
    return _out(k_ref_per_min * np.exp(-(Ea_kJ * 1e3 / R_GAS) * (1.0 / T_K - 1.0 / T_ref_K)))


def rate_ratio(T_low_C: float, T_high_C: float, Ea_kJ: float) -> float:
    """두 온도의 속도 비 k(T_high)/k(T_low) (무차원).

    """
    return float(k_chem(T_high_C, 1.0, Ea_kJ, T_ref_C=T_low_C))


def t_eq_min(temperature_C: ArrayLike, time_min: ArrayLike, Ea_kJ: float = 130.0,
             T_ref_C: float = 80.0) -> ArrayLike:
    """기준 온도(기본 80°C) 환산 가열 시간 t_eq (분).

    t_eq = t·exp[−Ea/R·(1/T − 1/T_ref)], T는 K. 식품 살균의 F값과 같은 개념이다.
    모델 A에서 λ − λ₀ = k(T_ref)·t_eq이므로 DDI 하나가 알려주는 양은 이것뿐이다.

    Args:
        temperature_C: 가열 온도 (°C).
        time_min: 가열 시간 (분).
        Ea_kJ: 활성화 에너지 (kJ/mol).
        T_ref_C: 기준 온도 (°C).
    """
    ratio = np.asarray(k_chem(temperature_C, 1.0, Ea_kJ, T_ref_C))
    return _out(ratio * np.asarray(time_min, dtype=float))


def lam_A(temperature_C: ArrayLike, time_min: ArrayLike, lam0: float,
          k_ref_per_min: float, Ea_kJ: float, T_ref_C: float = 80.0) -> ArrayLike:
    """모델 A (일정 속도): λ = λ₀ + k(T)·t [nt⁻¹].

    """
    k = k_chem(temperature_C, k_ref_per_min, Ea_kJ, T_ref_C)
    return _out(lam0 + np.asarray(k) * np.asarray(time_min, dtype=float))


def lam_A_residual(temperature_C: ArrayLike, time_min: ArrayLike, lam0: float,
                   k_ref_per_min: float, Ea_kJ: float, tau_min: float,
                   T_ref_C: float = 80.0) -> ArrayLike:
    """모델 A' (잔열): λ = λ₀ + k(T)·(t + τ) [nt⁻¹].

    τ는 급랭하지 않아 식는 동안 더해진 유효 가열 시간(분). 절편/기울기 = τ로
    온도와 무관하다. 근거: 01 문서 4-1 A', 3-2 보조 판별.
    """
    k = k_chem(temperature_C, k_ref_per_min, Ea_kJ, T_ref_C)
    return _out(lam0 + np.asarray(k) * (np.asarray(time_min, dtype=float) + tau_min))


def lam_B(temperature_C: ArrayLike, time_min: ArrayLike, lam0: float,
          k_ref_per_min: float, Ea_kJ: float, X: TempParam, q_per_min: TempParam,
          T_ref_C: float = 80.0) -> ArrayLike:
    """모델 B (다단계): λ = λ₀ + k(T)·t + X(T)·(1 − e^(−q(T)·t)) [nt⁻¹].

    X(nt⁻¹), q(min⁻¹)는 스칼라 또는 온도(°C)의 함수. 전체 형태는 해석과 합성
    데이터에만 쓰고 판정은 B'로 한다. 근거: 01 문서 4-1 B (Kistler 2017).
    """
    t = np.asarray(time_min, dtype=float)
    k = np.asarray(k_chem(temperature_C, k_ref_per_min, Ea_kJ, T_ref_C))
    X_v = _resolve(X, temperature_C)
    q_v = _resolve(q_per_min, temperature_C)
    return _out(lam0 + k * t + X_v * (1.0 - np.exp(-q_v * t)))


def lam_B_intercept(temperature_C: ArrayLike, time_min: ArrayLike, lam0: float,
                    k_ref_per_min: float, Ea_kJ: float, X: TempParam,
                    T_ref_C: float = 80.0) -> ArrayLike:
    """모델 B' (절편 모델): λ ≈ λ₀ + X(T) + k(T)·t, t ≥ 15분 [nt⁻¹].

    B에서 q가 커서 초기 분해가 첫 측정 시간 안에 끝난 극한. 판정은 이 형태로 한다.
    """
    t = np.asarray(time_min, dtype=float)
    k = np.asarray(k_chem(temperature_C, k_ref_per_min, Ea_kJ, T_ref_C))
    X_v = _resolve(X, temperature_C)
    return _out(lam0 + X_v + k * t)
