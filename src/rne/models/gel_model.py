"""전기영동 측정 모델 (핵 DNA, 두 가닥 절단).

"""
from __future__ import annotations

import numpy as np


def gel_mass_density(x_bp: np.ndarray, r_per_bp: float, ladder_amp: float = 0.0,
                     spacing_bp: float = 190.0, ladder_width_bp: float = 15.0) -> np.ndarray:
    """길이별 질량 밀도 C(x)를 계산한다 (상대값, 무차원).

    C(x) ∝ x·e^(−r·x) + a·Σ_{j≥1} G(x; d·j).
    - 무작위 절단 항 x·e^(−rx)는 봉우리(x = 1/r)의 높이가 1이 되도록 r·e를 곱해 정규화한다
    - G(x; μ)는 중심 μ, 표준편차 w = ladder_width_bp·√j 인 가우스이며 봉우리 높이는 1이다.
      j번째 띠의 폭이 √j로 커지는 것은 연결 부위 길이 변동이 누적된다고 본 가정이다
    - 따라서 a(ladder_amp)는 "무작위 절단 봉우리 대비 사다리 띠 높이"다

    Args:
        x_bp: 조각 길이 (bp) 배열.
        r_per_bp: 절단 밀도 r (bp⁻¹).
        ladder_amp: 사다리 세기 a (A 모델이면 0).
        spacing_bp: 사다리 간격 d (bp).
        ladder_width_bp: 첫 띠의 가우스 표준편차 (bp).
    """
    x = np.asarray(x_bp, dtype=float)
    smooth = r_per_bp * np.e * x * np.exp(-r_per_bp * x)
    if ladder_amp == 0.0:
        return smooth
    j_max = int(np.ceil(np.max(x) / spacing_bp)) + 3
    ladder = np.zeros_like(x)
    for j in range(1, j_max + 1):
        w = ladder_width_bp * np.sqrt(j)
        ladder += np.exp(-0.5 * ((x - spacing_bp * j) / w) ** 2)
    return smooth + ladder_amp * ladder
