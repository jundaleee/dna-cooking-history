"""StepOnePlus "Export → Results"를 qpcr_long 형식으로 읽는다.

"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rne.common.schema import QPCR_LONG_COLUMNS, TARGETS

#: 대조 웰(NTC 말고 프라이머 없음·믹스만·물만)에 기기에서 넣는 표적 이름. 분석 표적이 아니므로 그대로 둔다
CONTROL_TARGETS = frozenset({"CTRL"})
from rne.qpcr._io import (
    _norm,
    find_column,
    is_ct_header,
    normalize_well,
    read_export_table,
    read_metadata,
    to_float,
)

#: StepOne v2.x가 개별 열로 내는 경고 이름 (값은 Y/N 또는 빈칸)
FLAG_COLUMNS = (
    "BADROX", "EXPFAIL", "NOAMP", "HIGHSD", "OUTLIERRG", "AMPNC", "AMPSCORE", "BLFAIL",
    "CQCONF", "CRTAMP", "CRTNOISE", "DRNMIN", "MAXCT", "NOISE", "NOSIGNAL", "OFFSCALE",
    "PRFDROP", "PRFLOW", "SPIKE", "THOLDFAIL", "TAQMANAMB", "EXPHIGHSD", "MTP",
)

#: 저장소 안 기본 대응표 위치 (P7에서 만든다)
DEFAULT_ALIAS_PATH = Path(__file__).resolve().parents[3] / "templates" / "target_alias.json"

#: 대응표 파일이 없을 때 쓰는 최소 대응 (대소문자 무시)
BUILTIN_ALIAS = {
    "S104": "S104", "104": "S104", "104bp": "S104", "104 bp": "S104", "short": "S104",
    "L586": "L586", "586": "L586", "586bp": "L586", "586 bp": "L586", "long": "L586",
}


def load_target_alias(path: str | Path | None = None) -> dict[str, str]:
    """표적 이름 대응표를 읽는다. path가 None이면 templates/target_alias.json, 없으면 내장표."""
    p = Path(path) if path is not None else DEFAULT_ALIAS_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return data
    if path is not None:
        raise FileNotFoundError(p)
    return dict(BUILTIN_ALIAS)


def map_target(name, alias: dict[str, str]) -> str:
    """기기에 입력된 Target Name을 `S104`/`L586`으로. 모르면 경고하고 그대로 둔다."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip()
    if s in TARGETS:
        return s
    if s.upper() in CONTROL_TARGETS:  # 대조 칸의 표적 이름. 경고 없이 그대로
        return s.upper()
    low = {_norm(k): v for k, v in alias.items()}
    if _norm(s) in low:
        return low[_norm(s)]
    warnings.warn(f"표적 이름 '{s}'가 대응표에 없다. 그대로 둔다 (templates/target_alias.json에 추가)")
    return s


def _flag_value_true(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().upper()
    return s in {"Y", "YES", "TRUE", "1", "X"}


def read_steponeplus(path, target_alias=None, plate_id: str | None = None) -> pd.DataFrame:
    """StepOne Software v2.x "Export → Results" 파일을 qpcr_long 형식으로 읽는다.

    - 형식: `.xls`/`.xlsx`(시트 "Results"), `.txt`/`.csv`(`[Results]` 구역 또는 표만)
    - 앞쪽 `*` 메타데이터 줄은 건너뛰고 첫 칸이 `Well`인 머리줄을 찾는다
    - Ct 열 이름이 `CT`/`Ct`/`Cт`(키릴 문자) 등으로 달라도 찾는다. `Undetermined` → NaN, `undetermined=True`
    - 경고: `Flag`/`Comments` 한 열의 글자, 또는 `BADROX`·`EXPFAIL` 같은 개별 Y/N 열 → `instrument_flags`에 `;`로
    - `target_alias`: dict 또는 JSON 경로. None이면 `templates/target_alias.json`
    - `Task`(UNKNOWN/STANDARD/NTC)는 `task` 열로 덧붙인다(qpcr_long 약속 열 밖의 추가 열)
    - `melt_peaks`: Tm1, Tm2… 열 중 값이 있는 개수. Tm 열이 없으면 NaN
    - `tech_rep`: 같은 (시료, 표적) 안에서 웰 순서대로 1, 2, 3…
    - `plate_id`: None이면 메타데이터의 Experiment File Name, 그것도 없으면 파일 이름
    """
    path = Path(path)
    if target_alias is None or isinstance(target_alias, (str, Path)):
        alias = load_target_alias(target_alias)
    else:
        alias = dict(target_alias)

    raw = read_export_table(path, "Results")
    meta = read_metadata(path)

    well_col = find_column(raw, "Well Position", required=False)
    if well_col is None or raw[well_col].isna().all():
        well_col = find_column(raw, "Well")
    sample_col = find_column(raw, "Sample Name", "Sample")
    target_col = find_column(raw, "Target Name", "Target", "Detector Name", "Detector")
    ct_cols = [c for c in raw.columns if is_ct_header(c)]
    if not ct_cols:
        raise KeyError(f"{path}: Ct 열을 찾지 못했다 (열: {list(raw.columns)})")
    ct_col = ct_cols[0]
    task_col = find_column(raw, "Task", required=False)
    qty_col = find_column(raw, "Quantity", required=False)
    flag_text_cols = [c for c in raw.columns if _norm(c) in {"flag", "flags", "comments", "comment"}]
    flag_bool_cols = [c for c in raw.columns if str(c).strip().upper() in FLAG_COLUMNS]
    tm_cols = [c for c in raw.columns if re.fullmatch(r"tm\s*\d+", _norm(c))]

    out = pd.DataFrame()
    out["well"] = [normalize_well(v) for v in raw[well_col]]
    out["sample_id"] = raw[sample_col].map(lambda v: "" if v is None else str(v).strip())
    out["target"] = raw[target_col].map(lambda v: map_target(v, alias))
    ct_raw = raw[ct_col]
    out["undetermined"] = ct_raw.map(lambda v: isinstance(v, str) and v.strip().lower() == "undetermined")
    out["ct"] = ct_raw.map(to_float)
    out.loc[out["undetermined"], "ct"] = np.nan
    out["task"] = raw[task_col].map(lambda v: "" if v is None else str(v).strip().upper()) if task_col else ""
    qty = raw[qty_col].map(to_float) if qty_col else pd.Series(np.nan, index=raw.index)
    # 표준곡선 웰이 아니면 Quantity는 기기가 계산한 추정치이므로 std_quantity에 넣지 않는다
    is_std = out["task"].eq("STANDARD") if task_col else qty.notna()
    out["std_quantity"] = np.where(is_std, qty, np.nan)
    if tm_cols:
        tm = raw[tm_cols].apply(lambda col: col.map(to_float))
        out["melt_peaks"] = tm.notna().sum(axis=1).astype(float)
        out["tm1"] = tm.iloc[:, 0]
    else:
        out["melt_peaks"] = np.nan
    flags = []
    for i in range(len(raw)):
        fl: list[str] = []
        for c in flag_text_cols:
            v = raw.at[i, c]
            if v is None:
                continue
            for tok in re.split(r"[;,\s]+", str(v).strip()):
                tok = tok.strip().upper()
                if tok in FLAG_COLUMNS and tok not in fl:
                    fl.append(tok)
        for c in flag_bool_cols:
            name = str(c).strip().upper()
            if _flag_value_true(raw.at[i, c]) and name not in fl:
                fl.append(name)
        flags.append(";".join(fl))
    out["instrument_flags"] = flags

    # 빈 웰(시료 이름·표적 모두 없음, Ct 없음)은 버린다
    empty = out["sample_id"].eq("") & out["target"].eq("") & out["ct"].isna() & ~out["undetermined"]
    out = out[~empty].reset_index(drop=True)

    pid = plate_id or meta.get("Experiment File Name") or path.stem
    pid = Path(str(pid).replace("\\", "/")).stem if pid else path.stem
    out.insert(0, "plate_id", pid)
    out["tech_rep"] = out.groupby(["sample_id", "target"]).cumcount() + 1
    ordered = QPCR_LONG_COLUMNS + [c for c in out.columns if c not in QPCR_LONG_COLUMNS]
    out = out[ordered]
    out.attrs["metadata"] = meta
    return out


def passive_reference(path) -> str | None:
    """메타데이터의 Passive Reference 설정(예: "ROX", ""). 없으면 None."""
    meta = read_metadata(path)
    for k, v in meta.items():
        if _norm(k) == "passive reference":
            return v
    return None
