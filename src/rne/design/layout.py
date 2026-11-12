"""96칸 판 배치표 만들기: 두 구간 짝을 이웃 칸에, 시료 위치를 조건과 무관하게, 재료를 하나씩 뺀 대조 칸을 흩어 놓는다.

- 한 짝 = 같은 줄의 (홀수 열: 짧은 구간, 짝수 열: 긴 구간). 판 하나에 짝 48개
- 대조 짝: 주형 없음(NTC), 프라이머 없음(같은 짝의 두 칸에 서로 다른 시료 주형), 믹스만, 물만
- 시료 위치를 무작위로 여러 번 뽑아, 줄·열이 온도·시간을 설명하는 비율(η²)의 최댓값이 가장 작은 배치를 고른다.
  대조 짝은 여러 줄·열에 흩어지게 한다
판을 돌리기 전에 만들고 사전 고정 기록에 넣는다. 실측 값은 쓰지 않는다.

    python -m rne.design.layout --temps 50,60,70,75,80 --times 15,30,60 --reps 3,2,3 --out plate_layout.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rne.design.damage import design

ROWS = "ABCDEFGH"


def eta2(x, g) -> float:
    """g로 설명되는 x의 분산 비."""
    x, g = np.asarray(x, float), np.asarray(g)
    tot = ((x - x.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    return float(sum((g == u).sum() * (x[g == u].mean() - x.mean()) ** 2 for u in np.unique(g)) / tot)


def make_layout(temps_C, times_min, reps, n_ntc_pairs: int = 4, n_noprimer_pairs: int = 2,
                n_mixonly_pairs: int = 1, n_water_pairs: int = 1, targets=("S104", "L586"),
                seed: int = 0, n_search: int = 20000) -> tuple[pd.DataFrame, float]:
    """배치표(well, target, kind, sample_id, temperature_C, time_min, replicate, mix, template)와 최대 η²."""
    T, t, n = design(temps_C, times_min, reps)
    samples = [(float(Ti), float(ti), k + 1) for Ti, ti, ni in zip(T, t, n) for k in range(int(ni))]
    rng = np.random.default_rng(seed)
    kinds = [("sample", s) for s in samples] + [("ntc", None)] * n_ntc_pairs
    for _ in range(n_noprimer_pairs):
        a, b = rng.choice(len(samples), 2, replace=False)
        kinds.append(("noprimer", (samples[a], samples[b])))
    kinds += [("mixonly", None)] * n_mixonly_pairs + [("water", None)] * n_water_pairs
    if len(kinds) > 48:
        raise ValueError(f"짝 {len(kinds)}개는 96칸 판(짝 48개)에 들어가지 않는다")
    ns, nc = len(samples), len(kinds) - len(samples)
    slots = [(r, c) for r in range(8) for c in range(6)]
    best, score = None, np.inf
    for _ in range(n_search):
        place = [slots[i] for i in rng.permutation(48)[:len(kinds)]]
        rr = np.array([p[0] for p in place[:ns]])
        cc = np.array([p[1] for p in place[:ns]])
        TT = np.array([s[0] for _, s in kinds[:ns]])
        tt = np.array([s[1] for _, s in kinds[:ns]])
        ctrl = place[ns:]
        if nc and (len({p[0] for p in ctrl}) < min(6, nc) or len({p[1] for p in ctrl}) < min(5, nc)):
            continue
        v = max(eta2(rr, TT), eta2(cc, TT), eta2(rr, tt), eta2(cc, tt))
        if v < score:
            best, score = place, v
    rows = []
    for (r, c), (kind, s) in zip(best, kinds):
        for side, target in enumerate(targets):
            d = dict(well=f"{ROWS[r]}{2 * c + 1 + side}", target=target, kind=kind, sample_id="", temperature_C="",
                     time_min="", replicate="", mix=f"{target} 믹스", template="시료")
            if kind == "sample":
                Ti, ti, k = s
                d.update(sample_id=f"C-{Ti:03.0f}-{ti:g}-{k}", temperature_C=Ti, time_min=ti, replicate=k)
            elif kind == "ntc":
                d.update(template="멸균수")
            elif kind == "noprimer":
                Ti, ti, k = s[side]
                d.update(target="CTRL", sample_id=f"C-{Ti:03.0f}-{ti:g}-{k}", temperature_C=Ti, time_min=ti,
                         replicate=k, mix="프라이머 없는 믹스")
            elif kind == "mixonly":
                d.update(target="CTRL", mix="프라이머 없는 믹스", template="멸균수")
            else:
                d.update(target="CTRL", mix="없음", template="멸균수")
            rows.append(d)
    df = pd.DataFrame(rows)
    df["_r"], df["_c"] = df.well.str[0], df.well.str[1:].astype(int)
    return df.sort_values(["_r", "_c"]).drop(columns=["_r", "_c"]).reset_index(drop=True), float(score)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="96칸 판 배치표 만들기")
    ap.add_argument("--temps", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--reps", required=True, help="정수, 시간마다, 또는 조건마다(쉼표로)")
    ap.add_argument("--ntc", type=int, default=4, help="주형 없음 짝 수")
    ap.add_argument("--noprimer", type=int, default=2, help="프라이머 없음 짝 수")
    ap.add_argument("--mixonly", type=int, default=1)
    ap.add_argument("--water", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plate_layout.csv")
    a = ap.parse_args(argv)
    nums = lambda s: [float(x) for x in s.split(",") if x.strip()]  # noqa: E731
    reps = [int(x) for x in nums(a.reps)]
    df, score = make_layout(nums(a.temps), nums(a.times), reps[0] if len(reps) == 1 else reps,
                            a.ntc, a.noprimer, a.mixonly, a.water, seed=a.seed)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{a.out}: {len(df)}칸 {df.kind.value_counts().to_dict()}, 줄·열의 최대 η² {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
