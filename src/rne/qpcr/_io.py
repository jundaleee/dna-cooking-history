"""StepOne Software 내보내기 파일에서 표 하나를 찾아 읽는 공통 도구.

StepOne Software v2.x의 "Export"는 같은 내용을 여러 모양으로 낸다.

- `.xls`/`.xlsx`: 시트 하나에 내보내기 종류 하나(시트 이름 "Results", "Amplification Data",
  "Multicomponent Data", "Melt Curve Raw Data"). 여러 종류를 한 파일로 내보내면 시트가 여러 개
- `.txt`/`.csv`: 탭(또는 쉼표) 구분. 여러 종류를 한 파일로 내보내면 `[Results]` 같은
  대괄호 줄로 구역이 나뉜다
- 둘 다 맨 앞에 `* Experiment File Name = ...` 같은 `*`로 시작하는 메타데이터 줄이 있을 수 있다

여기서는 "구역/시트 이름"과 "머리줄 첫 칸이 Well"이라는 두 단서로 표를 찾는다.
모든 값은 문자열(또는 엑셀 숫자 그대로)로 돌려주고, 형 변환은 각 reader가 한다.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
TEXT_SUFFIXES = {".txt", ".csv", ".tsv"}
TEXT_ENCODINGS = ("utf-8-sig", "cp949", "cp1252", "latin-1")


def _norm(s: object) -> str:
    """머리줄 비교용: 소문자, 앞뒤 공백·BOM 제거, 연속 공백 하나로."""
    return re.sub(r"\s+", " ", str(s).replace("﻿", "").strip().lower())


def read_metadata(path: str | Path) -> dict[str, str]:
    """`* 키 = 값` 메타데이터 줄(또는 실측 .xls의 `키 | 값` 두 칸 줄)을 dict로. 없으면 빈 dict.

    예: `* Passive Reference = ROX` → {"Passive Reference": "ROX"}.
    판정 0단계에서 기준 염료 설정을 확인하는 데 쓴다.
    """
    rows = _raw_rows(path, section=None)
    meta: dict[str, str] = {}
    for row in rows:
        if not row:
            continue
        first = str(row[0]).strip() if row[0] is not None else ""
        if _norm(first) == "well":
            break  # 머리줄부터는 표
        if not first.startswith("*"):
            # 실측 .xls(StepOne v2.3, 2차 판): `*` 없이 두 칸 "Passive Reference | ROX"
            if first and len(row) > 1 and row[1] not in (None, "") and all(
                    v in (None, "") or (isinstance(v, float) and np.isnan(v)) for v in row[2:]):
                meta[first] = str(row[1]).strip()
            continue
        text = first.lstrip("*").strip()
        if "=" in text:
            k, v = text.split("=", 1)
            meta[k.strip()] = v.strip()
        elif len(row) > 1 and row[1] not in (None, ""):
            meta[text.rstrip("=").strip()] = str(row[1]).strip()
    return meta


def read_export_table(
    path: str | Path,
    section: str | None,
    header_first: str = "well",
) -> pd.DataFrame:
    """내보내기 파일에서 `section` 구역(또는 시트)의 표를 찾아 DataFrame으로.

    - `section`: "Results", "Amplification Data" 등. 대소문자·공백 무시. None이면 첫 표
    - 머리줄: 첫 칸이 `header_first`("Well")인 첫 줄
    - 머리줄 다음의 빈 줄 또는 다음 구역 줄(`[...]`)에서 표가 끝난다
    없으면 ValueError.
    """
    rows = _raw_rows(path, section=section)
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0] is not None and _norm(row[0]) == header_first:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{path}: '{section or '첫 표'}'에서 '{header_first}' 머리줄을 찾지 못했다")
    header = [str(h).replace("﻿", "").strip() if h is not None else "" for h in rows[header_idx]]
    while header and header[-1] == "":
        header.pop()
    body = []
    for row in rows[header_idx + 1:]:
        if not row or all(v is None or str(v).strip() == "" for v in row):
            break
        first = str(row[0]).strip() if row[0] is not None else ""
        if first.startswith("[") and first.endswith("]"):
            break
        vals = list(row[: len(header)]) + [None] * max(0, len(header) - len(row))
        body.append(vals)
    df = pd.DataFrame(body, columns=_dedupe(header))
    return df


def _dedupe(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _raw_rows(path: str | Path, section: str | None) -> list[list]:
    """파일을 줄(칸 목록)의 목록으로. 엑셀이면 맞는 시트, 텍스트면 맞는 구역."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        try:
            return _excel_rows(path, section)
        except Exception as exc:  # StepOne이 텍스트를 .xls 이름으로 내는 경우 대비
            try:
                return _text_rows(path, section)
            except Exception:
                raise exc
    return _text_rows(path, section)


def _excel_rows(path: Path, section: str | None) -> list[list]:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*xlrd engine is deprecated.*")
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine, dtype=object)
    names = list(sheets)
    chosen = None
    if section is not None:
        target = _norm(section)
        for n in names:
            if _norm(n) == target:
                chosen = n
                break
        if chosen is None:
            for n in names:
                if target in _norm(n) or _norm(n) in target:
                    chosen = n
                    break
        if chosen is None:
            # 시트 하나에 대괄호 구역이 있는 경우
            for n in names:
                rows = _df_to_rows(sheets[n])
                sub = _slice_section(rows, section)
                if sub is not None:
                    return sub
            if len(names) == 1:
                chosen = names[0]
            else:
                raise ValueError(f"{path}: '{section}' 시트가 없다 (시트: {names})")
    else:
        chosen = names[0]
    rows = _df_to_rows(sheets[chosen])
    sub = _slice_section(rows, section) if section is not None else None
    return sub if sub is not None else rows


def _df_to_rows(df: pd.DataFrame) -> list[list]:
    rows = []
    for rec in df.itertuples(index=False):
        rows.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in rec])
    return rows


def _text_rows(path: Path, section: str | None) -> list[list]:
    raw = path.read_bytes()
    text = None
    for enc in TEXT_ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1은 항상 성공
        raise ValueError(f"{path}: 인코딩을 알 수 없다")
    lines = text.splitlines()
    sample = "\n".join(l for l in lines if l and not l.startswith("*"))[:5000]
    delim = "\t" if "\t" in sample else ","
    rows = [_split(l, delim) for l in lines]
    if section is not None:
        sub = _slice_section(rows, section)
        if sub is not None:
            return sub
        has_sections = any(r and str(r[0]).strip().startswith("[") for r in rows)
        if has_sections:
            raise ValueError(f"{path}: '[{section}]' 구역이 없다")
    return rows


def _split(line: str, delim: str) -> list:
    if delim == ",":
        import csv

        return next(csv.reader([line])) if line else []
    return line.split("\t") if line else []


def _slice_section(rows: list[list], section: str) -> list[list] | None:
    """`[section]` 줄 다음부터 다음 `[...]` 줄 전까지. 구역 표시가 없으면 None."""
    target = _norm(section)
    start = None
    for i, row in enumerate(rows):
        if not row or row[0] is None:
            continue
        first = str(row[0]).strip()
        if first.startswith("[") and first.endswith("]"):
            if start is not None:
                return rows[start:i]
            if _norm(first[1:-1]) == target:
                start = i + 1
    if start is not None:
        return rows[start:]
    return None


def find_column(df: pd.DataFrame, *candidates: str, required: bool = True) -> str | None:
    """후보 이름(대소문자·공백 무시) 중 df에 있는 첫 열 이름. 없으면 None 또는 KeyError."""
    norm = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in norm:
            return norm[_norm(cand)]
    if required:
        raise KeyError(f"열을 찾지 못했다: {candidates} (있는 열: {list(df.columns)})")
    return None


def is_ct_header(name: str) -> bool:
    """Ct 열인지. StepOne은 `CT`, `Ct`, `Cт`(키릴 т), 인코딩이 깨지면 `CÑ‚` 등으로 쓴다.

    `Ct Mean`, `Ct SD`, `Ct Threshold`는 제외한다.
    """
    n = _norm(name)
    return bool(re.fullmatch(r"c(t|т|ñ‚|ñ|\?|.{1,2})", n)) and n not in {"cq mean"} and not n.startswith("co")


def to_float(v) -> float:
    """숫자로. 빈칸·`Undetermined`·변환 불가 → NaN. 천 단위 쉼표 제거."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s == "" or s.lower() in {"undetermined", "nan", "n/a", "na", "-"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


WELL_RE = re.compile(r"^\s*([A-Ha-h])\s*0*(\d{1,2})\s*$")


def normalize_well(v, n_cols: int = 12) -> str:
    """웰 이름을 `A1` 형식으로. 숫자(1–96)면 행 우선 순서로 바꾼다."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    m = WELL_RE.match(s)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    if s.isdigit():
        i = int(s) - 1
        return f"{'ABCDEFGH'[i // n_cols]}{i % n_cols + 1}"
    warnings.warn(f"웰 이름을 해석하지 못했다: {s!r}")
    return s


def well_row_col(well: str) -> tuple[str, int]:
    """`B7` → ("B", 7)."""
    m = WELL_RE.match(str(well))
    if not m:
        return ("", -1)
    return (m.group(1).upper(), int(m.group(2)))
