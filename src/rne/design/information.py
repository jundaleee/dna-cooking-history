"""실험 전에 계산하는 정보량과 오류율 하한.

측정값(DDI) | 조건 ~ 정규(신호, σ²)이므로 상호정보량은 I = h(DDI) − ½log₂(2πeσ²)이고, 정규 혼합의 엔트로피를
수치 적분으로 구한다(근사 없음). K개 조건을 가리는 어떤 모델도 오류율을 (H − I − 1)/log₂(K − 1)보다 낮출 수 없다
(파노 부등식). 합격·불합격 두 갈래는 이상적 판정자의 최소 오류율(베이즈 오류)을 직접 구한다.
"""
from __future__ import annotations

import numpy as np

from rne.design.damage import EA_KJ, damage_ddi, design


def exact_mi(means, weights, sigma: float) -> float:
    """I(조건; 측정)(비트). means: 조건별 신호 평균, weights: 조건별 비중, sigma: 측정 잡음 SD."""
    means = np.asarray(means, float)
    w = np.asarray(weights, float) / np.sum(weights)
    grid = np.linspace(means.min() - 8 * sigma, means.max() + 8 * sigma, 6001)
    dx = grid[1] - grid[0]
    dens = np.zeros_like(grid)
    for s in range(0, len(means), 2000):
        m, ww = means[s:s + 2000], w[s:s + 2000]
        dens += (ww[None, :] * np.exp(-0.5 * ((grid[:, None] - m[None, :]) / sigma) ** 2)).sum(1)
    dens /= sigma * np.sqrt(2 * np.pi)
    pos = dens > 0
    h = -np.sum(dens[pos] * np.log2(dens[pos])) * dx
    return float(max(h - 0.5 * np.log2(2 * np.pi * np.e * sigma ** 2), 0.0))


def entropy_bits(weights) -> float:
    """조건의 엔트로피 H(비트)."""
    p = np.asarray(weights, float) / np.sum(weights)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def fano_lower_bound(I_bits: float, weights) -> float:
    """K개 조건을 가릴 때 어떤 판정자도 넘을 수 없는 오류율 하한(파노 부등식)."""
    K = int(np.sum(np.asarray(weights) > 0))
    if K < 3:
        raise ValueError("파노 하한은 조건이 셋 이상일 때 쓴다. 두 갈래는 binary_min_error")
    return float(max((entropy_bits(weights) - I_bits - 1) / np.log2(K - 1), 0.0))


def binary_min_error(means, weights, passed, sigma: float) -> tuple[float, float]:
    """합격(passed=True)·불합격 두 갈래의 최소 오류율과, 정보 없이 많은 쪽으로 답할 때의 오류율."""
    means = np.asarray(means, float)
    w = np.asarray(weights, float) / np.sum(weights)
    passed = np.asarray(passed, bool)
    x = np.linspace(means.min() - 8 * sigma, means.max() + 8 * sigma, 20001)
    dx = x[1] - x[0]

    def dens(mask):
        m, ww = means[mask], w[mask]
        return (ww[None, :] * np.exp(-0.5 * ((x[:, None] - m[None, :]) / sigma) ** 2)).sum(1) / (sigma * np.sqrt(2 * np.pi))

    err = float(np.minimum(dens(passed), dens(~passed)).sum() * dx)
    return err, float(min(w[passed].sum(), w[~passed].sum()))


def design_information(temps_C, times_min, reps, D60: float, sigma: float = 0.42, world: str = "M1",
                       ea_kJ: float = EA_KJ, pass_at=None, shift=None, t_lag_min: float = 0.0) -> dict:
    """설계 하나의 정보량 요약.

    pass_at: (온도, 시간)을 주면 그 가열량 이상을 합격으로 보고 두 갈래 최소 오류율도 낸다.
    shift: 조건별로 더할 계통 오차(cycle, 길이 = 조건 수). t_lag_min: 가열 시간에서 뺄 승온 지연(분).
    """
    T, t, n = design(temps_C, times_min, reps)
    m = damage_ddi(T, np.maximum(t - t_lag_min, 0.1), D60, world, ea_kJ)
    if shift is not None:
        m = m + np.asarray(shift, float)
    I = exact_mi(m, n, sigma)
    out = {"conditions": int(len(T)), "samples": int(n.sum()), "H_bits": entropy_bits(n), "I_bits": I,
           "chance_error": float(1 - n.max() / n.sum())}
    if len(T) >= 3:
        out["fano_min_error"] = fano_lower_bound(I, n)
    if pass_at is not None:
        crit = float(damage_ddi(np.array([float(pass_at[0])]), np.array([float(pass_at[1])]), D60, world, ea_kJ)[0])
        passed = damage_ddi(T, t, D60, world, ea_kJ) >= crit - 1e-12
        err, chance = binary_min_error(m, n, passed, sigma)
        out.update(pass_fraction=float(n[passed].sum() / n.sum()), binary_min_error=err, binary_chance_error=chance)
    return out


def gaussian_bits(rho: float) -> float:
    """상관 ρ로 낸 정규 근사 I = −½log₂(1 − ρ²)(비트). 직선 관계를 가정하므로 정보를 낮게 잡을 수 있다."""
    r = min(max(float(rho), 0.0), 0.999999)
    return float(-0.5 * np.log2(1.0 - r * r))


def ksg_bits(pred, y, k: int = 3, seed: int = 0) -> float:
    """예측과 참값 사이의 상호정보량, KSG 추정량(비트). 정보가 없어도 0.03비트 안팎이 나올 수 있다."""
    from sklearn.feature_selection import mutual_info_regression
    return float(mutual_info_regression(np.asarray(pred, float).reshape(-1, 1), np.asarray(y, float),
                                        n_neighbors=k, random_state=seed)[0] / np.log(2))
