"""시료별 DDI = Ct(586) − Ct(104).

- 기술 반복 평균
- 중도절단: 원시 신호에서 다시 계산한 Ct가 **40 cycle까지 증폭이 없을 때만**(미검출 또는
  Ct ≥ censor_at) 숫자로 채우지 않는다: ddi 비움, ddi_censored=True, ddi_lower = censor_at − ct_short.
  35 ≤ Ct < 40은 값을 그대로 쓰고 `long_beyond_loq`/`short_beyond_loq`로 표시만 한다(예전 loq=35 경로는
  censor_at=35로 되살릴 수 있다)
- 효율 E: 인자 → 표준곡선 → 웰별 효율 중앙값 → 1.0 순. 어느 것을 썼는지 qc_flag에 남긴다
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rne.common.schema import DDI_COLUMNS, LOQ_CT, MAX_CYCLES
from rne.qpcr.qc import standard_curve, well_role


def _pick_efficiency(given, qpcr_long, target, well_eff) -> tuple[float, str]:
    if given is not None and np.isfinite(given):
        return float(given), "given"
    sc = standard_curve(qpcr_long, target)
    if sc["ok"]:
        tag = "std_curve" if sc["in_range"] else "std_curve_out_of_range"
        return float(sc["efficiency"]), tag
    if well_eff is not None and len(well_eff):
        e = well_eff[(well_eff["target"] == target)]["efficiency"].dropna()
        if len(e):
            return float(e.median()), "well_median"
    return 1.0, "assumed_1.0"


def _join(*parts) -> str:
    toks = []
    for p in parts:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        for t in str(p).split(";"):
            t = t.strip()
            if t and t not in toks:
                toks.append(t)
    return ";".join(toks)


def compute_ddi(qpcr_long: pd.DataFrame, sample_map: pd.DataFrame, efficiency_short=None, efficiency_long=None,
                loq: float = LOQ_CT, well_eff: pd.DataFrame | None = None,
                censor_at: float = float(MAX_CYCLES)) -> pd.DataFrame:
    """qpcr_long과 sample_map으로 ddi.csv 형식 표를 만든다 (시료 하나가 한 줄).

    - 대상: sample_map에서 group이 ntc/std가 아닌 시료. qpcr_long에 웰이 없으면 `no_qpcr`
    - ct_short/ct_long: 미검출이 아닌 기술 반복의 평균. Ct ≥ censor_at(기본 40)은 미검출로 본다
    - 586 bp: 모든 반복이 미검출 → 중도절단. 일부 반복만 미검출이어도 중도절단하고
      `long_rep_undetermined`를 남긴다(남은 반복 평균은 실제보다 작게 치우치므로)
    - 104 bp가 없으면(웰 없음 또는 전부 미검출) `no_short`, ddi·ddi_lower 비움
    - 평균 Ct ≥ loq(35)이면 `short_beyond_loq`/`long_beyond_loq` 표시만(값은 그대로)
    - 효율: `eff_short=<출처>`, `eff_long=<출처>` (given, std_curve, std_curve_out_of_range, well_median, assumed_1.0)
    - well_eff: well_efficiency 결과(웰별 효율 중앙값 대체용, 선택)
    """
    role = well_role(qpcr_long)
    q = qpcr_long[role == "unknown"]
    e_s, src_s = _pick_efficiency(efficiency_short, qpcr_long, "S104", well_eff)
    e_l, src_l = _pick_efficiency(efficiency_long, qpcr_long, "L586", well_eff)
    sm = sample_map[~sample_map["group"].astype(str).isin(["ntc", "std"])]
    rows = []
    for s in sm.itertuples(index=False):
        flags = []
        ws = q[(q["sample_id"] == s.sample_id) & (q["target"] == "S104")]
        wl = q[(q["sample_id"] == s.sample_id) & (q["target"] == "L586")]
        if len(ws) == 0 and len(wl) == 0:
            flags.append("no_qpcr")
        det_s = ws["ct"].dropna()
        det_s = det_s[det_s < censor_at]
        det_l = wl["ct"].dropna()
        det_l = det_l[det_l < censor_at]
        ct_s = float(det_s.mean()) if len(det_s) else np.nan
        ct_l = float(det_l.mean()) if len(det_l) else np.nan
        ddi, cens, lower = np.nan, False, np.nan
        if not np.isfinite(ct_s):
            flags.append("no_short")
        else:
            if ct_s >= loq:
                flags.append("short_beyond_loq")
            long_partial = len(wl) > 0 and len(det_l) < len(wl)
            if len(wl) == 0:
                flags.append("no_long")
            elif not np.isfinite(ct_l) or long_partial:
                cens = True
                lower = censor_at - ct_s
                if long_partial and np.isfinite(ct_l):
                    flags.append("long_rep_undetermined")
            else:
                ddi = ct_l - ct_s
                if ct_l >= loq:
                    flags.append("long_beyond_loq")
        if len(ws) == 1 or len(wl) == 1:
            flags.append("single_tech_rep")
        flags += [f"eff_short={src_s}", f"eff_long={src_l}"]
        rows.append(dict(sample_id=s.sample_id, temperature_C=s.temperature_C, time_min=s.time_min,
                         replicate=s.replicate, extraction_batch=s.extraction_batch, group=s.group,
                         ct_short=ct_s, ct_long=ct_l, ddi=ddi, ddi_censored=cens, ddi_lower=lower,
                         efficiency_short=e_s, efficiency_long=e_l,
                         qc_flag=_join(getattr(s, "qc_flag", ""), *flags)))
    return pd.DataFrame(rows, columns=DDI_COLUMNS)


def with_recomputed_ct(qpcr_long: pd.DataFrame, recomputed: pd.DataFrame, ct_column: str) -> pd.DataFrame:
    """qpcr_long의 기기 Ct를 원시 신호에서 다시 계산한 Ct로 바꾼다.

    recomputed: well과 ct_column 열(예: dual_ct의 `ct_rox`/`ct_norox`, recompute_ct의 `ct_recomputed`).
    기기 Ct는 `ct_instrument`로 남기고, 재계산에서 미검출이면 ct = NaN, undetermined = True.
    재계산 표에 없는 웰은 ct = NaN으로 두고 `ct_source`에 `missing`.
    """
    q = qpcr_long.copy()
    m = recomputed.drop_duplicates("well").set_index("well")[ct_column]
    q["ct_instrument"] = q["ct"]
    q["ct"] = q["well"].map(m).astype(float)
    q["undetermined"] = q["ct"].isna()
    q["ct_source"] = np.where(q["well"].isin(m.index), ct_column, "missing")
    return q
