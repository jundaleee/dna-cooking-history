"""원신호 Ct: 기기가 기준 염료(ROX)로 나누기 전의 형광(Multicomponent의 SYBR)으로 Ct를 다시 낸다.

기기의 Ct는 형광을 기준 염료로 나눈 값(ΔRn)에 문턱을 적용한다. 기준 염료가 형광을 따라 오르면 나눈 값이 지워져
Ct가 크게 늦어질 수 있다(예: 원신호 Ct 11.1인 칸이 기기 Ct 39.4). 정의:
- 칸마다 기준선 = 지정 cycle 구간(기본 1–3)의 원신호 중앙값, ΔF = 원신호 − 기준선
- 문턱 = 판 전체 칸의 고원(최대 ΔF) 중앙값 × threshold_frac(기본 10 %)
- Ct = ΔF가 문턱을 넘는 cycle(로그 선형 보간, 다시 내려가지 않는 마지막 교차)

    python -m rne.qpcr.raw_ct --export <판 .xls> --out raw_ct.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rne.qpcr.curves import _crossing
from rne.qpcr.raw import read_multicomponent


def raw_signal_ct(multicomponent: pd.DataFrame, dye: str = "SYBR", baseline_cycles=(1, 3),
                  threshold_frac: float = 0.10) -> pd.DataFrame:
    """칸별 원신호 Ct. 반환 열: well, baseline, plateau_dF, threshold, ct_raw(넘지 못하면 NaN)."""
    d = multicomponent[multicomponent["dye"].astype(str).str.upper() == dye.upper()]
    per = []
    for w, g in d.groupby("well", sort=False):
        g = g.sort_values("cycle")
        c, y = g["cycle"].to_numpy(float), g["fluorescence"].to_numpy(float)
        base = float(np.median(y[(c >= baseline_cycles[0]) & (c <= baseline_cycles[1])]))
        per.append((str(w), c, y - base, base))
    plateaus = np.array([dF.max() for _, _, dF, _ in per])
    thr = threshold_frac * float(np.median(plateaus[plateaus > 0])) if np.any(plateaus > 0) else np.nan
    rows = [{"well": w, "baseline": base, "plateau_dF": float(dF.max()), "threshold": thr,
             "ct_raw": _crossing(c, dF, thr) if np.isfinite(thr) else np.nan} for w, c, dF, base in per]
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="원신호(Multicomponent SYBR)로 Ct 다시 내기")
    ap.add_argument("--export", required=True, help="StepOne/StepOnePlus 내보내기(.xls 또는 .txt)")
    ap.add_argument("--out", default="raw_ct.csv")
    ap.add_argument("--baseline", default="1,3", help="기준선 cycle 구간")
    ap.add_argument("--frac", type=float, default=0.10, help="문턱 = 판 고원 중앙값 × 이 값")
    a = ap.parse_args(argv)
    b = tuple(float(x) for x in a.baseline.split(","))
    out = raw_signal_ct(read_multicomponent(a.export), baseline_cycles=b, threshold_frac=a.frac)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{a.out}: {len(out)}칸, 문턱 {out.threshold.iloc[0]:.0f}, Ct 없음 {int(out.ct_raw.isna().sum())}칸")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
