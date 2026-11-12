"""공유 상수와 파일 스키마.

이 파일은 공유 파일이다. 바꾸려면 리포트의 "공유 파일 변경 제안"에 적는다.
"""
from __future__ import annotations

import pandas as pd

# --- 표적 (프라이머: 통합본 6장) ---
S104_BP: int = 104
L586_BP: int = 586
DELTA_BP: int = L586_BP - S104_BP  # 482

# --- 실험 설계 (NanoDrop 기록 기준, 인수인계 v2 2-2) ---
TEMPERATURES_C: tuple[float, ...] = (50.0, 60.0, 70.0, 75.0, 80.0)
TIMES_MIN: tuple[float, ...] = (15.0, 30.0, 60.0)
N_REP: int = 6

# --- qPCR ---
LOQ_CT: float = 35.0
MAX_CYCLES: int = 40
TARGETS: tuple[str, str] = ("S104", "L586")

# --- 동역학 ---
R_GAS: float = 8.314  # J/(mol·K)
EA_RANGE_KJ: tuple[float, float] = (110.0, 150.0)
EA_REF_KJ: float = 130.0
H1_RATIO_THRESHOLD: float = 32.0  # 50→80°C 속도 비, Ea 110 kJ/mol 하한

# --- 전기영동 ---
GEL_WINDOW_MIN_BP: float = 1000.0
GEL_RESOLUTION_MAX_BP: float = 10000.0
LADDER_SPACING_RANGE_BP: tuple[float, float] = (170.0, 220.0)

# --- 파일 스키마  ---
SAMPLE_MAP_COLUMNS = [
    "sample_id", "group", "temperature_C", "time_min", "replicate",
    "extraction_batch", "marinade", "qc_flag",
]
QPCR_LONG_COLUMNS = [
    "plate_id", "well", "sample_id", "target", "ct", "undetermined",
    "tech_rep", "std_quantity", "melt_peaks", "instrument_flags",
]
DDI_COLUMNS = [
    "sample_id", "temperature_C", "time_min", "replicate", "extraction_batch", "group",
    "ct_short", "ct_long", "ddi", "ddi_censored", "ddi_lower",
    "efficiency_short", "efficiency_long", "qc_flag",
]
GEL_PROFILE_COLUMNS = ["gel_id", "lane", "sample_id", "position_px", "intensity", "saturated"]
GEL_LADDER_COLUMNS = ["gel_id", "lane", "band_bp", "position_px"]
GEL_SUMMARY_COLUMNS = [
    "gel_id", "lane", "sample_id", "valid", "saturated", "window_min_bp",
    "r_hat_per_bp", "r_se_per_bp", "peak_bp", "mean_bp", "r_lower_bound_only",
    "ladder_amp", "ladder_spacing_bp", "rnase_checked", "notes",
]
GROUPS = ("heat", "raw", "marinade", "commercial", "ntc", "std")


def check_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """df에 약속된 열이 모두 있는지 확인한다. 없으면 ValueError."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: 필요한 열이 없다: {missing}")
