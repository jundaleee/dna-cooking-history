"""검정력: 같은 설계로 가상 실험을 되풀이했을 때 가열 효과를 우연과 구별하는 비율.

가상 실험 하나: 시료를 만들고(`simulate_samples`), 조건 단위 교차검증(같은 조건의 반복을 통째로 뺌)으로 선형 회귀가
log 가열량을 예측한 정보량(정규 근사)을 잰다. 참값을 조건째로 섞은 순열 기준선의 95 % 분위수를 넘으면 '구별함'.
조건이 적으면(대략 6개 미만) 조건째 순열의 경우의 수와 학습에 남는 조건이 모자라 검정력이 낮게 나온다.
가상 수치는 설계 점검용이다.
"""
from __future__ import annotations

import numpy as np

from rne.design.damage import design, expand, simulate_samples, theta
from rne.design.information import gaussian_bits


def _corr(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _one_experiment(T, t, cond, D60, sigma, seed, n_perm, world):
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneGroupOut
    rng = np.random.default_rng(seed)
    X = simulate_samples(T, t, D60, sigma, rng, world)
    y = np.log(theta(T, t))
    pred = np.empty(len(y))
    for tr, te in LeaveOneGroupOut().split(X, groups=cond):
        pred[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    I = gaussian_bits(_corr(y, pred))
    _, first, inv = np.unique(cond, return_index=True, return_inverse=True)
    null = [gaussian_bits(_corr(rng.permutation(y[first])[inv], pred)) for _ in range(n_perm)]
    return I > np.quantile(null, 0.95)


def detection_power(temps_C, times_min, reps, D60: float, sigma: float = 0.42, n_sim: int = 200,
                    n_perm: int = 199, seed: int = 0, world: str = "M1", n_jobs: int = 1) -> float:
    """설계(온도·시간·반복)에서 가열 효과를 우연과 구별할 확률."""
    T, t, cond = expand(*design(temps_C, times_min, reps))
    seeds = [seed * 100003 + k for k in range(n_sim)]
    if n_jobs == 1:
        res = [_one_experiment(T, t, cond, D60, sigma, s, n_perm, world) for s in seeds]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_one_experiment)(T, t, cond, D60, sigma, s, n_perm, world)
                                      for s in seeds)
    return float(np.mean(res))
