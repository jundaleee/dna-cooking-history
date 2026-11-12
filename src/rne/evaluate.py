"""평가 단위에 맞춘 교차검증과 배치 안 순열 검정.

같은 조건의 반복이나 같은 추출 배치의 시료는 값이 비슷하다. 시료를 하나씩 빼면 시험 시료와 거의 같은 값이 학습에
남아 성능이 부풀려진다(데이터 누수). 조건(또는 로트·배치)째로 빼서 평가하고, 우연과의 비교는 라벨을 같은 배치
안에서만 섞는 순열 검정으로 한다(배치 차이가 우연 성능에 포함되도록).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_cv_predict(model, X, y, groups=None) -> np.ndarray:
    """groups가 같은 시료를 통째로 빼는 교차검증 예측. groups=None이면 시료 하나씩 빼기."""
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
    X, y = np.asarray(X, float), np.asarray(y, float)
    splits = LeaveOneOut().split(X) if groups is None else LeaveOneGroupOut().split(X, groups=np.asarray(groups))
    pred = np.empty(len(y))
    for tr, te in splits:
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def within_batch_permutation(model, X, y, groups, batches, n_perm: int = 999, seed: int = 0,
                             n_jobs: int = 1) -> dict:
    """관측 R²와, 라벨을 배치 안에서만 섞어 같은 교차검증을 되풀이한 R² 분포로 낸 순열 p."""
    y = np.asarray(y, float)
    batches = np.asarray(batches)
    obs = r2(y, grouped_cv_predict(model, X, y, groups))

    def one(k):
        rng = np.random.default_rng(seed * 1000003 + k)
        yp = y.copy()
        for b in np.unique(batches):
            idx = np.flatnonzero(batches == b)
            yp[idx] = rng.permutation(y[idx])
        return r2(yp, grouped_cv_predict(model, X, yp, groups))

    if n_jobs == 1:
        null = np.array([one(k) for k in range(n_perm)])
    else:
        from joblib import Parallel, delayed
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(k) for k in range(n_perm)))
    return {"R2": obs, "p": float((1 + np.sum(null >= obs)) / (1 + n_perm)),
            "null_median": float(np.median(null)), "null_q95": float(np.quantile(null, 0.95)), "n_perm": n_perm}


def compare_schemes(model, X, y, condition, batch, n_perm: int = 999, seed: int = 0, n_jobs: int = 1) -> pd.DataFrame:
    """세 평가 단위(시료 하나씩, 조건 단위, 추출 배치 단위)를 나란히: R², 평균 절대 오차, 배치 안 순열 p."""
    y = np.asarray(y, float)
    rows = []
    for name, groups in (("시료 하나씩 빼기", None), ("조건 단위", condition), ("추출 배치 단위", batch)):
        p = grouped_cv_predict(model, X, y, groups)
        row = {"평가": name, "R2": r2(y, p), "MAE": float(np.mean(np.abs(y - p)))}
        if name != "추출 배치 단위":   # 배치째로 빼면 배치 안 순열의 기준선이 정의되지 않는다
            row["순열_p"] = within_batch_permutation(model, X, y, groups, batch, n_perm, seed, n_jobs)["p"]
        rows.append(row)
    return pd.DataFrame(rows)
