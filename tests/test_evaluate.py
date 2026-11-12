"""평가 단위 시험: 반복끼리 닮은 자료에서 시료 하나씩 빼기는 온도와 무관한 입력도 좋아 보이게 한다."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from rne.evaluate import compare_schemes, grouped_cv_predict, r2, within_batch_permutation


def _leaky_data(seed=0):
    rng = np.random.default_rng(seed)
    cond = np.repeat(np.arange(15), 6)
    temp = np.repeat(np.tile([50, 60, 70, 75, 80], 3), 6).astype(float)
    batch = (temp >= 75).astype(int)
    shared = rng.normal(0, 1, 15)[cond]                 # 같은 조건의 반복이 공유하는 값(온도와 무관)
    X = np.column_stack([shared + rng.normal(0, 0.05, cond.size), rng.normal(0, 1, cond.size)])
    return X, temp, cond, batch


def test_grouped_cv_predict_shapes():
    X, y, cond, _ = _leaky_data()
    p = grouped_cv_predict(RandomForestRegressor(n_estimators=20, random_state=0), X, y, cond)
    assert p.shape == y.shape and np.isfinite(r2(y, p))


def test_condition_unit_removes_leak():
    X, y, cond, _ = _leaky_data()
    model = RandomForestRegressor(n_estimators=30, random_state=0)
    loo = r2(y, grouped_cv_predict(model, X, y, None))
    grp = r2(y, grouped_cv_predict(model, X, y, cond))
    assert loo > grp


def test_permutation_p_range_and_table():
    X, y, cond, batch = _leaky_data(1)
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    res = within_batch_permutation(model, X, y, cond, batch, n_perm=19, seed=0)
    assert 0 < res["p"] <= 1 and res["n_perm"] == 19
    t = compare_schemes(model, X, y, cond, batch, n_perm=9)
    assert list(t["평가"]) == ["시료 하나씩 빼기", "조건 단위", "추출 배치 단위"]
