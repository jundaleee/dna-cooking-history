"""StepOne 원시 곡선 내보내기 읽기: Amplification Data, Multicomponent Data, Melt Curve Raw Data.

1차 판(11/4)은 기기 판단(BADROX, EXPFAIL, 자동 기준선)만 남아 원인을 가를 수 없었다.
이번에는 원시 곡선을 직접 읽어 P3에서 다시 계산한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from rne.qpcr._io import _norm, find_column, normalize_well, read_export_table, to_float
from rne.qpcr.results import load_target_alias, map_target

AMPLIFICATION_COLUMNS = ["well", "target", "cycle", "rn", "delta_rn"]
MULTICOMPONENT_COLUMNS = ["well", "cycle", "dye", "fluorescence"]
MELT_COLUMNS = ["well", "temperature_C", "fluorescence", "derivative"]


def _well_series(raw: pd.DataFrame) -> pd.Series:
    col = find_column(raw, "Well Position", required=False)
    if col is None or raw[col].isna().all():
        col = find_column(raw, "Well")
    return raw[col].map(normalize_well)


def read_amplification(path, target_alias=None) -> pd.DataFrame:
    """Amplification Data 내보내기 → 열 `well, target, cycle, rn, delta_rn` (웰·사이클마다 한 줄).

    - `Rn`: 정규화 형광(SYBR/ROX), `Delta Rn`: 기기가 자동 기준선을 뺀 값
    - 표적 이름은 read_steponeplus와 같은 대응표로 바꾼다(target_alias)
    - 단위: cycle은 사이클 번호(1부터)
    """
    raw = read_export_table(path, "Amplification Data")
    alias = load_target_alias(target_alias) if target_alias is None or isinstance(target_alias, str) else target_alias
    out = pd.DataFrame({"well": _well_series(raw)})
    tcol = find_column(raw, "Target Name", "Target", "Detector Name", "Detector", required=False)
    if tcol is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            names = raw[tcol].map(lambda v: "" if v is None else str(v).strip())
            mapping = {n: map_target(n, alias) for n in names.unique()}
        out["target"] = names.map(mapping)
    else:
        out["target"] = ""
    out["cycle"] = raw[find_column(raw, "Cycle")].map(to_float).astype("Int64").astype(float)
    out["rn"] = raw[find_column(raw, "Rn")].map(to_float)
    dcol = find_column(raw, "Delta Rn", "ΔRn", "dRn", required=False)
    out["delta_rn"] = raw[dcol].map(to_float) if dcol else np.nan
    return out[AMPLIFICATION_COLUMNS].sort_values(["well", "cycle"]).reset_index(drop=True)


def read_multicomponent(path, max_cycle: float | None = "auto") -> pd.DataFrame:
    """Multicomponent Data 내보내기 → 긴 형식 `well, cycle, dye, fluorescence`.

    내보내기는 염료가 열(`SYBR`, `ROX` …)인 넓은 표다. `Well`, `Cycle`, `Well Position` 밖의
    숫자 열을 모두 염료로 본다. 염료 이름은 대문자로(예: `ROX`).
    BADROX 진단(rox_diagnostics)의 입력이다.

    실측 .xls(2차 판)는 증폭 40 cycle 뒤에 멜트 읽기까지 이어서 "Cycle" 1–157로 낸다. 멜트 구간을 섞으면
    ROX 진단이 틀어지므로 max_cycle="auto"이면 같은 파일의 Amplification Data 최대 cycle까지만 남긴다
    (Amplification Data가 없으면 자르지 않음). 숫자를 주면 그 cycle까지, None이면 자르지 않는다.
    """
    raw = read_export_table(path, "Multicomponent Data")
    if max_cycle == "auto":
        try:
            amp_raw = read_export_table(path, "Amplification Data")
            max_cycle = float(amp_raw[find_column(amp_raw, "Cycle")].map(to_float).max())
        except (ValueError, KeyError):
            max_cycle = None
    wells = _well_series(raw)
    ccol = find_column(raw, "Cycle")
    skip = {_norm(c) for c in ("Well", "Well Position", "Cycle", "Sample Name", "Target Name")}
    dye_cols = [c for c in raw.columns if _norm(c) not in skip and c != ""]
    frames = []
    for c in dye_cols:
        vals = raw[c].map(to_float)
        if vals.notna().sum() == 0:
            continue
        frames.append(pd.DataFrame({
            "well": wells, "cycle": raw[ccol].map(to_float),
            "dye": str(c).strip().upper(), "fluorescence": vals,
        }))
    if not frames:
        return pd.DataFrame(columns=MULTICOMPONENT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    if max_cycle is not None and np.isfinite(max_cycle):
        out = out[out["cycle"] <= max_cycle]
    return out[MULTICOMPONENT_COLUMNS].sort_values(["well", "dye", "cycle"]).reset_index(drop=True)


def _read_melt_region(path) -> pd.DataFrame | None:
    """실측 .xls의 넓은 멜트 시트(웰 한 줄, Reading 1…N 열): "Melt Region Temperature Data",
    "Melt Region Normalized Data", "Melt Region Derivative Data" → 긴 형식. 시트가 없으면 None."""
    try:
        temp = read_export_table(path, "Melt Region Temperature Data")
    except ValueError:
        return None
    # 구역 표시 없는 텍스트 파일은 어떤 이름으로 찾아도 첫 표가 돌아온다 → 넓은 Reading 열이 있을 때만 인정
    if sum(_norm(c).startswith("reading ") for c in temp.columns) < 3:
        return None

    def _long(df, value):
        wcol = find_column(df, "Well Location", "Well Position", required=False) or find_column(df, "Well")
        rcols = [c for c in df.columns if _norm(c).startswith("reading")]
        m = df[[wcol] + rcols].melt(id_vars=wcol, var_name="reading", value_name=value)
        m["well"] = m[wcol].map(normalize_well)
        m["reading"] = m["reading"].map(lambda c: int(str(c).split()[-1]))
        m[value] = m[value].map(to_float)
        return m[["well", "reading", value]]

    out = _long(temp, "temperature_C")
    for sheet, col in (("Melt Region Normalized Data", "fluorescence"), ("Melt Region Derivative Data", "derivative")):
        try:
            out = out.merge(_long(read_export_table(path, sheet), col), on=["well", "reading"], how="left")
        except ValueError:
            out[col] = np.nan
    return out


def read_melt(path) -> pd.DataFrame:
    """Melt Curve Raw Data 내보내기 → `well, temperature_C, fluorescence, derivative`.

    - `Derivative` 열이 있으면 그대로(기기는 −dF/dT를 양수 봉우리로 낸다), 없으면 웰마다 −dF/dT를 계산
    - 실측 .xls(StepOne v2.3, 2차 판)는 "Melt Curve Raw Data" 대신 넓은 "Melt Region … Data" 시트를 낸다.
      그 시트가 있으면 그것을 먼저 읽는다(Normalized → fluorescence, Derivative → derivative)
    - 단위: temperature_C는 °C
    """
    region = _read_melt_region(path)
    if region is not None:
        out = region.sort_values(["well", "temperature_C"]).reset_index(drop=True)
        if out["derivative"].isna().all():
            out["derivative"] = negative_derivative(out)
        return out[MELT_COLUMNS]
    raw = read_export_table(path, "Melt Curve Raw Data")
    out = pd.DataFrame({"well": _well_series(raw)})
    out["temperature_C"] = raw[find_column(raw, "Temperature", "Temp")].map(to_float)
    out["fluorescence"] = raw[find_column(raw, "Fluorescence", "Rn")].map(to_float)
    dcol = find_column(raw, "Derivative", "-Rn'", "Derivative Reporter", required=False)
    if dcol is not None:
        out["derivative"] = raw[dcol].map(to_float)
    else:
        out["derivative"] = np.nan
    out = out.sort_values(["well", "temperature_C"]).reset_index(drop=True)
    if out["derivative"].isna().all():
        out["derivative"] = negative_derivative(out)
    return out[MELT_COLUMNS]


def negative_derivative(melt: pd.DataFrame) -> pd.Series:
    """웰마다 −dF/dT (중앙 차분). melt는 well, temperature_C, fluorescence 열."""
    res = pd.Series(np.nan, index=melt.index)
    for _, g in melt.groupby("well", sort=False):
        g = g.sort_values("temperature_C")
        if len(g) < 3:
            continue
        d = -np.gradient(g["fluorescence"].to_numpy(float), g["temperature_C"].to_numpy(float))
        res.loc[g.index] = d
    return res
