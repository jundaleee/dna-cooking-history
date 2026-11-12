"""실험 설계와 손상 신호: 실험 전 정보량·검정력 계산에 쓰는 손상 모형.

- 가열량 Θ = t·exp[−(Ea/R)(1/T − 1/T_ref)] (기준 온도 T_ref에서의 등가 시간, 분)
- 손상 신호 D60: 80 °C에서 60분 가열했을 때 늘어나는 DDI(cycle). 신호의 모양은 `rne.synth.fast.delta_lambda`,
  크기는 D60으로 맞춘다
- DDI = Ct(긴 구간) − Ct(짧은 구간). 두 구간의 효율이 같으면 주형 양이 지워진다
가상 수치는 설계 점검용이며 실측 결과가 아니다.
"""
from __future__ import annotations

import numpy as np

from rne.common.schema import R_GAS

EA_KJ = 130.0       # 이중가닥 DNA 탈퓨린화(Lindahl & Nyberg 1972, 31 ± 2 kcal/mol)
T_REF_C = 80.0      # 가열량의 기준 온도
E_AMP = 0.95        # 두 구간의 증폭 효율(같다고 둠)


def theta(T_C, t_min, ea_kJ: float = EA_KJ, T_ref_C: float = T_REF_C) -> np.ndarray:
    """가열량 Θ(분): 기준 온도 T_ref에서의 등가 가열 시간."""
    T = np.asarray(T_C, float) + 273.15
    return np.asarray(t_min, float) * np.exp(-(ea_kJ * 1e3 / R_GAS) * (1.0 / T - 1.0 / (T_ref_C + 273.15)))


def theta_from_profile(time_min, temp_C, ea_kJ: float = EA_KJ, T_ref_C: float = T_REF_C) -> float:
    """측정한 온도 곡선 T(t)로 낸 가열량 Θ = ∫exp[−(Ea/R)(1/T(t) − 1/T_ref)]dt(분, 사다리꼴 적분).
    탐침으로 잰 시료 속 온도로 계산해 물 온도로 낸 Θ와 비교하면 승온 지연의 크기를 알 수 있다."""
    t = np.asarray(time_min, float)
    T = np.asarray(temp_C, float) + 273.15
    rate = np.exp(-(ea_kJ * 1e3 / R_GAS) * (1.0 / T - 1.0 / (T_ref_C + 273.15)))
    return float(np.sum((rate[1:] + rate[:-1]) / 2 * np.diff(t)))


def damage_ddi(T_C, t_min, D60: float, world: str = "M1", ea_kJ: float = EA_KJ) -> np.ndarray:
    """가열이 만드는 DDI 변화(cycle). 80 °C 60분에서 D60이 되도록 맞춘다. world는 rne.synth.fast의 손상 모형 이름."""
    from rne.synth.fast import default_params, delta_lambda
    p = default_params()
    p["ea_kJ"] = ea_kJ
    ref = float(delta_lambda(world, np.array([80.0]), np.array([60.0]), p)[0])
    return D60 * delta_lambda(world, np.asarray(T_C, float), np.asarray(t_min, float), p) / ref


def design(temps_C, times_min, reps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """조건 표(온도 × 시간)와 조건마다의 반복 수.

    reps: 정수(모든 조건 같음), 길이 len(times_min)인 목록(시간마다, 모든 온도에 같은 무늬),
    또는 길이 len(temps_C)·len(times_min)인 목록(조건마다, 온도 바깥 순서).
    반환: 조건별 온도, 시간, 반복 수(배열 셋, 길이 = 조건 수).
    """
    temps = [float(x) for x in temps_C]
    times = [float(x) for x in times_min]
    T = np.repeat(temps, len(times))
    t = np.tile(times, len(temps))
    r = np.atleast_1d(np.asarray(reps, float))
    if r.size == 1:
        n = np.full(T.size, r[0])
    elif r.size == len(times):
        n = np.tile(r, len(temps))
    elif r.size == T.size:
        n = r
    else:
        raise ValueError("reps는 정수, 시간 수, 또는 조건 수 길이여야 한다")
    return T, t, n


def expand(T, t, n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """조건 표를 시료 단위로 펼친다. 반환: 시료별 온도, 시간, 조건 번호."""
    n = np.asarray(n, int)
    idx = np.repeat(np.arange(len(T)), n)
    return np.asarray(T)[idx], np.asarray(t)[idx], idx


def simulate_samples(T, t, D60: float, sigma: float, rng: np.random.Generator, world: str = "M1",
                     sigma_ct: float = 0.25, sigma_logN: float = 0.8, c0: float = 20.0, ddi0: float = 3.0,
                     E: float = E_AMP) -> np.ndarray:
    """가상 시료의 [DDI, Ct(짧은), Ct(긴)]. 시료 하나의 DDI 잡음 SD가 sigma가 되도록 칸 잡음과 시료 잡음을 나눈다."""
    n = len(T)
    ln1e = np.log1p(E)
    lam = damage_ddi(T, t, D60, world) * ln1e / 482.0
    bio_sd = np.sqrt(max(sigma ** 2 - 2 * sigma_ct ** 2, 0.0))
    lnN = rng.normal(0, sigma_logN, n)
    ct_s = c0 + (lam * 104 - lnN) / ln1e + rng.normal(0, sigma_ct, n)
    ct_l = c0 + ddi0 + (lam * 586 - lnN) / ln1e + rng.normal(0, sigma_ct, n) + rng.normal(0, bio_sd, n)
    return np.column_stack([ct_l - ct_s, ct_s, ct_l])
