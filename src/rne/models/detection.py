"""검출 한계.

"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rne.common.schema import DELTA_BP, N_REP, TIMES_MIN


def s_tt(times_min: Sequence[float] = TIMES_MIN, n_rep: int = N_REP) -> float:
    """시간 설계의 제곱합 S_tt = Σ(t − t̄)² (min²). 각 시간에 n_rep개씩.

    """
    t = np.repeat(np.asarray(times_min, dtype=float), n_rep)
    return float(np.sum((t - t.mean()) ** 2))


def lod_60(sigma_ddi: float, times_min: Sequence[float] = TIMES_MIN, n_rep: int = N_REP) -> float:
    """시간 기울기로 잡을 수 있는 60분간 DDI 변화의 한계 (cycle).

    LoD₆₀ = 2·σ/√S_tt·60. σ는 같은 조건 반복 간 DDI 표준편차.
    """
    return float(2.0 * sigma_ddi / np.sqrt(s_tt(times_min, n_rep)) * 60.0)


def ddi_per_teq(k_ref_per_min: float, efficiency: float = 1.0) -> float:
    """80°C 환산 1분당 DDI 증가 b = 482·k(80)/ln(1 + E) (cycle/min).

    모델 A에서 DDI = DDI₀ + b·t_eq. 근거: 01 문서 0-1, 4-2.
    """
    return float(DELTA_BP * k_ref_per_min / np.log1p(efficiency))


def min_detectable_teq(sigma_ddi: float, slope_cycle_per_min: float,
                       times_min: Sequence[float] = TIMES_MIN, n_rep: int = N_REP) -> float:
    """5-2의 LoD₆₀을 t_eq 단위로 바꾼 값 (80°C 환산 분).

    min t_eq = LoD₆₀ / b. 이 설계(시간 목록 × n_rep)의 시간 기울기로 잡을 수 있는
    가장 작은 누적 가열량이다. 설계 계산이며 결과가 아니다. 근거: 01 문서 5-2, 0-2 '가'.

    Args:
        sigma_ddi: 같은 조건 반복 간 DDI 표준편차 (cycle).
        slope_cycle_per_min: b, 80°C 환산 1분당 DDI 증가 (cycle/min). ddi_per_teq로 계산.
    """
    if slope_cycle_per_min <= 0:
        return float("inf")
    return lod_60(sigma_ddi, times_min, n_rep) / slope_cycle_per_min


def teq_resolution_min(sigma_ddi: float, slope_cycle_per_min: float, n_rep: int = 1) -> float:
    """DDI 해상도를 t_eq로 바꾼 값 σ/(b·√n) (80°C 환산 분).

    t̂_eq = (DDI − DDI₀)/b의 표준편차. n_rep개 평균이면 √n으로 준다.
    """
    if slope_cycle_per_min <= 0:
        return float("inf")
    return float(sigma_ddi / (slope_cycle_per_min * np.sqrt(n_rep)))


def log_teq_sd_approx(t_eq: float, t_res: float) -> float:
    """log t̂_eq의 표준편차 근사 t_res/t_eq (델타 방법, 무차원).

    t_eq ≫ t_res일 때만 맞다. t_eq ≲ t_res이면 t̂_eq가 0 근처나 음수가 되어 로그가
    정의되지 않고, 오차는 하한을 어디에 자르느냐로 정해진다 (r1_resolution.md 2절).
    """
    return float(t_res / t_eq) if t_eq > 0 else float("inf")
