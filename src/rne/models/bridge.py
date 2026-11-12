"""겔과 qPCR의 다리 (Johnston 2022 식 (2)).

"""
from __future__ import annotations

import numpy as np

from rne.common.schema import L586_BP, S104_BP


def johnston_intact_fraction(fragment_bp: np.ndarray, mass: np.ndarray, target_bp: float) -> float:
    """길이 분포에서 길이 r인 표적이 온전히 들어 있는 비율 P_r (0–1).

    P_r = Σ_{f≥r} ((f − r + 1)/f)·C_f / Σ_f C_f. C_f는 길이 f인 조각들의 질량
    (염색 밝기 ∝ 질량). 모수가 없다. 근거: 01 문서 4-4, Johnston 2022 식 (2).

    Args:
        fragment_bp: 조각 길이 f (bp) 배열.
        mass: 각 길이의 질량 C_f (상대값, 음수 불가).
        target_bp: 표적 길이 r (bp).
    """
    f = np.asarray(fragment_bp, dtype=float)
    c = np.asarray(mass, dtype=float)
    if f.shape != c.shape:
        raise ValueError("fragment_bp와 mass의 길이가 다르다")
    if np.any(c < 0):
        raise ValueError("mass에 음수가 있다")
    total = c.sum()
    if total <= 0:
        raise ValueError("mass의 합이 0이다")
    keep = f >= target_bp
    w = (f[keep] - target_bp + 1.0) / f[keep]
    return float(np.sum(w * c[keep]) / total)


def ddi_johnston(fragment_bp: np.ndarray, mass: np.ndarray, efficiency: float = 1.0) -> float:
    """길이 분포에서 예측한 DDI^J = ln(P₁₀₄/P₅₈₆)/ln(1 + E) (cycle).

    """
    p_short = johnston_intact_fraction(fragment_bp, mass, S104_BP)
    p_long = johnston_intact_fraction(fragment_bp, mass, L586_BP)
    if p_long <= 0:
        return float("inf")
    return float(np.log(p_short / p_long) / np.log1p(efficiency))


def d_statistic(delta_ddi_mt: float, delta_ddi_j_nuc: float) -> float:
    """D = ΔDDI_mt − ΔDDI^J_nuc (cycle).

    두 Δ는 같은 배치 비가열 대조군 대비 가열 효과. 게놈 차이가 없으면 D ≥ 0이어야
    한다(겔은 한 가닥 손상을 못 봄). 판정(부트스트랩 상한 < 0)은 decide 모듈이 한다.
    """
    return float(delta_ddi_mt - delta_ddi_j_nuc)
