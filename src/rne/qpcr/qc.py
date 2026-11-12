"""판정 0단계(qPCR): 표준곡선과 판 유효성.

01 문서 5-3 0단계 qPCR 기준을 그대로 쓴다:
NTC Ct ≥ 35 또는 미검출, 표준곡선 E 90–110 %, BADROX 없음, 멜트 피크 하나.
기준값을 바꾸지 않는다. 바꿔야 하면 리포트의 "사람이 결정할 것"에 적는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rne.common.schema import LOQ_CT, TARGETS

#: 표준곡선 E 허용 범위
STD_EFFICIENCY_RANGE = (0.90, 1.10)
#: 표준곡선에 필요한 최소 농도 점 수
MIN_STD_POINTS = 3



def _as_str(col: pd.Series) -> pd.Series:
    """결측을 빈 글자로 바꾼 문자열 열. float(전부 NaN) 열에 fillna("")를 쓰면 판다스 다음 버전에서 오류가 된다(K9)."""
    return col.astype(object).where(col.notna(), "").astype(str)


def well_role(qpcr_long: pd.DataFrame) -> pd.Series:
    """웰 역할 `ntc`/`std`/`unknown`. `task` 열(NTC/STANDARD)을 먼저 보고, 없으면 시료 이름(NTC-…, STD-…)과 std_quantity."""
    task = _as_str(qpcr_long["task"]).str.upper() if "task" in qpcr_long else pd.Series("", index=qpcr_long.index)
    sid = _as_str(qpcr_long["sample_id"]).str.upper()
    role = pd.Series("unknown", index=qpcr_long.index)
    is_std = task.eq("STANDARD") | sid.str.startswith("STD")
    if "std_quantity" in qpcr_long:
        is_std |= qpcr_long["std_quantity"].notna() & ~task.isin(["UNKNOWN", "NTC"])
    role[is_std] = "std"
    role[task.isin(["NTC", "NEGATIVE CONTROL"]) | sid.str.startswith("NTC")] = "ntc"
    return role


def standard_curve(qpcr_long: pd.DataFrame, target: str) -> dict:
    """표적 하나의 표준곡선: Ct = slope · log10(양) + intercept (웰 단위 최소제곱).

    반환: target, slope, intercept, r2, efficiency(E = 10^(−1/slope) − 1), n_points(서로 다른 농도 수),
    n_wells, in_range(E가 90–110 % 안), ok(점이 MIN_STD_POINTS 이상이고 E가 계산됨), note
    """
    role = well_role(qpcr_long)
    d = qpcr_long[(role == "std") & (qpcr_long["target"] == target)]
    d = d[d["ct"].notna() & d["std_quantity"].notna() & (d["std_quantity"] > 0)]
    res = dict(target=target, slope=np.nan, intercept=np.nan, r2=np.nan, efficiency=np.nan,
               n_points=int(d["std_quantity"].nunique()), n_wells=int(len(d)), in_range=False, ok=False, note="")
    if res["n_points"] < MIN_STD_POINTS:
        res["note"] = f"농도 점 {res['n_points']}개 (< {MIN_STD_POINTS})"
        return res
    x = np.log10(d["std_quantity"].to_numpy(float))
    y = d["ct"].to_numpy(float)
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan
    eff = 10 ** (-1.0 / slope) - 1.0 if slope < 0 else np.nan
    res.update(slope=float(slope), intercept=float(icpt), r2=float(r2), efficiency=float(eff), ok=bool(np.isfinite(eff)))
    res["in_range"] = bool(np.isfinite(eff) and STD_EFFICIENCY_RANGE[0] <= eff <= STD_EFFICIENCY_RANGE[1])
    if not np.isfinite(eff):
        res["note"] = "기울기가 음수가 아님"
    return res


def _as_melt_summary(melt):
    if melt is None:
        return None
    if "n_peaks" in melt.columns:
        return melt
    from rne.qpcr.curves import melt_summary

    return melt_summary(melt)


def _as_rox(rox):
    if rox is None:
        return None
    if "rox_flags" in rox.columns:
        return rox
    from rne.qpcr.curves import rox_diagnostics

    return rox_diagnostics(rox)


def infer_plate_set(qpcr_long: pd.DataFrame) -> str:
    """판 종류 추정: 표준(STANDARD) 웰이 있을 때만 `dilution`(희석 세트), 아니면 `stock`(원액 세트).

    NTC가 있는 원액 판도 있으므로 NTC만으로는 희석 세트로 보지 않는다. 추정이므로 호출 쪽에서 plate_set을 직접
    주는 것이 우선이다.
    """
    role = well_role(qpcr_long)
    return "dilution" if (role == "std").any() else "stock"


def qc_plate(qpcr_long: pd.DataFrame, rox: pd.DataFrame | None = None, melt: pd.DataFrame | None = None,
             loq: float = LOQ_CT, plate_set: str | None = None) -> tuple[pd.DataFrame, dict]:
    """판 하나의 0단계 판정.

    판정은 세 갈래로 나눈다: `reasons`(위반 → 실패), `cautions`(확정 결정에 따라 "주의"로 진행), `warnings`(참고).
    - 결정 3: NTC·표준이 **없으면** 실패가 아니라 주의(희석 세트에서 보완). NTC가 **증폭**하거나 E가 범위 밖이면 실패
    - 결정 13: 원액 세트(plate_set='stock')의 BADROX·멜트 피크 이상은 주의. 희석 세트에서는 실패
    - 멜트 정보가 없으면 주의(확인 불가)
    - plate_set: 'stock' | 'dilution' | None(→ infer_plate_set으로 추정, summary에 기록)

    - rox: rox_diagnostics 결과 또는 Multicomponent 원시(well, cycle, dye, fluorescence). 없으면 기기 BADROX만 본다
    - melt: melt_summary 결과 또는 Melt 원시. 없으면 qpcr_long의 melt_peaks(Tm 열)를 쓴다
    반환:
    1. 웰 표: qpcr_long + role, n_peaks, rox_flags, qc_well_flags, qc_well_pass
       웰 플래그: `badrox`, `ntc_amplified`(NTC Ct < loq), `melt_multi`, `melt_none`(증폭됐는데 피크 없음), `melt_unknown`
    2. 판 요약 dict: valid(= reasons 없음), reasons, cautions, warnings, plate_set, ntc, standard_curves, n_wells,
       instrument_flag_counts
    ROX 원시 진단 플래그(rox_low 등)는 0단계 기준이 아니므로 warnings에만 적는다.
    """
    q = qpcr_long.copy()
    q["role"] = well_role(q)
    set_given = plate_set is not None
    plate_set = plate_set or infer_plate_set(q)
    if plate_set not in ("stock", "dilution"):
        raise ValueError(f"plate_set은 'stock' 또는 'dilution': {plate_set}")
    ms = _as_melt_summary(melt)
    if ms is not None:
        q = q.merge(ms[["well", "n_peaks"]], on="well", how="left")
    else:
        q["n_peaks"] = q["melt_peaks"] if "melt_peaks" in q else np.nan
    rx = _as_rox(rox)
    if rx is not None:
        q = q.merge(rx[["well", "rox_flags"]], on="well", how="left")
    else:
        q["rox_flags"] = np.nan

    flags_all = []
    for r in q.itertuples(index=False):
        fl = []
        inst = "" if pd.isna(r.instrument_flags) else str(r.instrument_flags)
        if "BADROX" in inst.upper():
            fl.append("badrox")
        amplified = pd.notna(r.ct)
        if r.role == "ntc" and amplified and r.ct < loq:
            fl.append("ntc_amplified")
        if r.role != "ntc" and amplified:
            if pd.isna(r.n_peaks):
                fl.append("melt_unknown")
            elif r.n_peaks >= 2:
                fl.append("melt_multi")
            elif r.n_peaks == 0:
                fl.append("melt_none")
        flags_all.append(";".join(fl))
    q["qc_well_flags"] = flags_all
    q["qc_well_pass"] = q["qc_well_flags"].eq("")

    reasons, warns, cautions = [], [], []
    soft = cautions if plate_set == "stock" else reasons  # 결정 13
    targets = [t for t in TARGETS if (q["target"] == t).any()]
    other = sorted(set(q["target"]) - set(TARGETS) - {""})
    if other:
        warns.append(f"대응표에 없는 표적 이름: {', '.join(other)} (data/templates/target_alias.json)")
    if not targets:
        reasons.append("S104·L586 표적 웰이 없다")
    ntc_info = {}
    for t in targets:
        n = q[(q["role"] == "ntc") & (q["target"] == t)]
        bad = n[n["ct"].notna() & (n["ct"] < loq)]
        ntc_info[t] = dict(n_wells=int(len(n)), n_amplified_below_loq=int(len(bad)),
                           min_ct=float(n["ct"].min()) if n["ct"].notna().any() else np.nan)
        if len(n) == 0:
            cautions.append(f"NTC 없음({t}) → 오염 확인 불가")
        elif len(bad):
            reasons.append(f"NTC 증폭({t}): {len(bad)}웰 Ct < {loq:g} (최소 {bad['ct'].min():.2f})")
    curves = {}
    for t in targets:
        sc = standard_curve(q, t)
        curves[t] = sc
        if not sc["ok"]:
            cautions.append(f"표준곡선 없음({t}): {sc['note']} → E는 웰별 곡선 효율")
        elif not sc["in_range"]:
            reasons.append(f"표준곡선 E 범위 밖({t}): E = {sc['efficiency'] * 100:.1f}% (기준 90–110%)")
    n_badrox = int(q["qc_well_flags"].str.contains("badrox").sum())
    if n_badrox:
        soft.append(f"BADROX 웰 {n_badrox}개")
    n_multi = int(q["qc_well_flags"].str.contains("melt_multi").sum())
    n_none = int(q["qc_well_flags"].str.contains("melt_none").sum())
    n_unk = int(q["qc_well_flags"].str.contains("melt_unknown").sum())
    if n_multi:
        soft.append(f"멜트 피크 2개 이상 웰 {n_multi}개")
    if n_none:
        soft.append(f"증폭됐는데 멜트 피크 없는 웰 {n_none}개")
    if n_unk:
        cautions.append(f"멜트 정보 없는 증폭 웰 {n_unk}개 (Melt Curve Raw Data 또는 Tm 열 필요)")
    if rx is not None:
        n_rox = int(_as_str(rx["rox_flags"]).ne("").sum())
        if n_rox:
            warns.append(f"ROX 원시 진단 플래그 웰 {n_rox}개 (0단계 기준 밖, 참고)")
    flag_counts = {}
    for s in _as_str(q["instrument_flags"]):
        for tok in [x for x in str(s).split(";") if x]:
            flag_counts[tok] = flag_counts.get(tok, 0) + 1
    if plate_set == "stock" and (n_badrox or n_multi or n_none):
        cautions.append("원액 세트: BADROX·멜트 이상은 0단계 주의로 진행")
    summary = dict(valid=len(reasons) == 0, reasons=reasons, cautions=cautions, warnings=warns, ntc=ntc_info,
                   plate_set=plate_set, plate_set_source="given" if set_given else "inferred(NTC·표준 유무)",
                   standard_curves=curves, n_wells=int(len(q)), instrument_flag_counts=flag_counts,
                   plate_id=str(q["plate_id"].iloc[0]) if len(q) and "plate_id" in q else "")
    return q, summary
