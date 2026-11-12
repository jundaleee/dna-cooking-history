"""비트 단위 오차 예산: 오차 원천마다 측정에 담긴 정보량이 얼마나 바뀌는지.

생명과학 쪽 오차(시료 잡음, 두 구간 효율 차 × 추출 배치 차, 승온 지연, 활성화 에너지, 신호 크기)를 같은 단위(비트)로
놓아, 무엇부터 줄일지 정한다. 계통 오차는 정보를 줄이기만 하지 않고, 가열과 무관한 차이를 정보처럼 보이게 해
정보량을 부풀릴 수도 있다. 가상 수치는 설계 점검용이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rne.design.damage import design
from rne.design.information import design_information


def error_budget(temps_C, times_min, reps, D60: float = 0.9, sigma: float = 0.42,
                 noise_levels=(0.30, 0.60, 0.84), batch_temps=(), batch_offsets=(-0.27, -0.18, 0.18, 0.27),
                 lags_min=(1.5, 3.5), eas_kJ=(112.0, 150.0), signals=(0.02, 0.3, 3.0)) -> pd.DataFrame:
    """오차 원천별 정보량 표. batch_temps: 다른 추출 배치에서 나온 온도(그 조건들에 batch_offsets를 더한다)."""
    T, _, _ = design(temps_C, times_min, reps)
    args = (temps_C, times_min, reps)
    base = design_information(*args, D60=D60, sigma=sigma)["I_bits"]
    rows = [("기준", f"D60 {D60}, σ {sigma}", base)]
    for s in noise_levels:
        rows.append(("시료 잡음", f"σ {s} cycle", design_information(*args, D60=D60, sigma=s)["I_bits"]))
    if len(batch_temps):
        in_batch = np.isin(T, [float(x) for x in batch_temps])
        for off in batch_offsets:
            I = design_information(*args, D60=D60, sigma=sigma, shift=np.where(in_batch, off, 0.0))["I_bits"]
            rows.append(("효율 차 × 추출 배치 차", f"{off:+.2f} cycle({', '.join(str(x) for x in batch_temps)} °C)", I))
    for lag in lags_min:
        rows.append(("승온 지연", f"유효 시간 − {lag}분",
                     design_information(*args, D60=D60, sigma=sigma, t_lag_min=lag)["I_bits"]))
    for ea in eas_kJ:
        rows.append(("활성화 에너지", f"{ea:g} kJ/mol", design_information(*args, D60=D60, sigma=sigma, ea_kJ=ea)["I_bits"]))
    for d in signals:
        rows.append(("신호 크기", f"D60 {d} cycle", design_information(*args, D60=d, sigma=sigma)["I_bits"]))
    out = pd.DataFrame(rows, columns=["오차 원천", "설정", "I_bits"])
    out["변화_bits"] = out.I_bits - base
    return out
