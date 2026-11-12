"""문헌 속도를 80°C 60분 DDI 증가로 환산한다.

공통 가정: 시간 비례(모델 A), 아레니우스 온도 환산, E = 1, ΔL = 482 bp.
문헌의 Ct 차이 Δcycle을 길이 차 ΔL_src에서 손상 밀도로 바꿀 때는
Δλ = Δcycle·ln(1 + E)/ΔL_src을 쓴다.
"""
from __future__ import annotations

import numpy as np

from rne.common.schema import DELTA_BP, EA_REF_KJ
from rne.models.kinetics import rate_ratio

# --- 원 조건 ---
OPENWETWARE_RATE_PER_S: float = 3e-11   # 37°C, pH 7.4
OPENWETWARE_T_C: float = 37.0
OPENWETWARE_EA_KJ: float = 121.0        # 원 조건 열의 Ea

HIRD_SHORT_BP: int = 81
HIRD_LONG_BP: int = 351
HIRD_BOIL_T_C: float = 100.0
HIRD_BOIL_TIME_MIN: float = 32.0
HIRD_BOIL_MAX_DIFF_CYCLE: float = 1.0   # 351·81 bp 차이 변화 1 cycle 이하로 가정
HIRD_AUTO_T_C: float = 121.0
HIRD_AUTO_TIME_MIN: float = 15.0
HIRD_AUTO_SHORT_SHIFT: float = 1.41     # 81 bp +1.41 cycle
HIRD_AUTO_LONG_SHIFT: float = 10.0      # 351 bp 약 +10 cycle

BRISCO_T_C: float = 99.0
BRISCO_LENGTH_BP: int = 200
BRISCO_TIMES_MIN: tuple[float, ...] = (0.0, 20.0, 30.0)
BRISCO_AF: tuple[float, ...] = (0.93, 0.48, 0.32)  # 200 bp 증폭 가능 분율 (05 문서 10번)


def ddi_increase(k_per_min: float, T_src_C: float, Ea_kJ: float, efficiency: float = 1.0,
                 T_target_C: float = 80.0, time_min: float = 60.0) -> float:
    """원 온도의 속도 k(T_src)를 목표 조건의 DDI 증가로 바꾼다 (cycle).

    ΔDDI = 482·k(T_src)·[k(T_target)/k(T_src)]·t / ln(1 + E).

    Args:
        k_per_min: 원 온도에서 nt당 분당 손상 (nt⁻¹·min⁻¹).
        T_src_C: 원 온도 (°C).
        Ea_kJ: 온도 환산에 쓸 활성화 에너지 (kJ/mol).
        efficiency: 증폭 효율 E.
        T_target_C, time_min: 목표 조건 (기본 80°C 60분).
    """
    k_target = k_per_min * rate_ratio(T_src_C, T_target_C, Ea_kJ)
    return float(DELTA_BP * k_target * time_min / np.log1p(efficiency))


def _rate_from_cycle_diff(diff_cycle: float, delta_len_bp: float, time_min: float,
                          efficiency: float = 1.0) -> float:
    """두 길이 표적의 Ct 차이 변화를 nt당 분당 손상 속도로 바꾼다 (nt⁻¹·min⁻¹)."""
    return float(diff_cycle * np.log1p(efficiency) / delta_len_bp / time_min)


def openwetware_rate_per_min() -> float:
    """문헌 속도 정리 페이지의 37°C 속도 3×10⁻¹¹/s를 분당으로 (nt⁻¹·min⁻¹)."""
    return OPENWETWARE_RATE_PER_S * 60.0


def hird_boil_rate_per_min(efficiency: float = 1.0) -> float:
    """Hird 2006 끓임 (100°C 32분, 351·81 bp 차이 ≤ 1 cycle)의 속도 상한."""
    return _rate_from_cycle_diff(HIRD_BOIL_MAX_DIFF_CYCLE, HIRD_LONG_BP - HIRD_SHORT_BP,
                                 HIRD_BOIL_TIME_MIN, efficiency)


def hird_autoclave_rate_per_min(efficiency: float = 1.0) -> float:
    """Hird 2006 오토클레이브 (121°C 15분, 81 bp +1.41, 351 bp +10)의 속도."""
    diff = HIRD_AUTO_LONG_SHIFT - HIRD_AUTO_SHORT_SHIFT
    return _rate_from_cycle_diff(diff, HIRD_LONG_BP - HIRD_SHORT_BP, HIRD_AUTO_TIME_MIN, efficiency)


def brisco_rate_per_min() -> float:
    """Brisco 2010 (99°C 0·20·30분, 200 bp AF 0.93·0.48·0.32)의 속도.

    λ(t) = −ln(AF)/200 (AF = e^(−λL), 05 문서 10번)을 세 시점에 대해 최소제곱 직선으로
    맞춘 기울기. 0분 값(0.93)이 가열 전 손상(λ₀)을 흡수한다.
    """
    t = np.asarray(BRISCO_TIMES_MIN)
    lam = -np.log(np.asarray(BRISCO_AF)) / BRISCO_LENGTH_BP
    return float(np.polyfit(t, lam, 1)[0])


def literature_predictions(Ea_kJ: float = EA_REF_KJ, efficiency: float = 1.0) -> dict[str, float]:
    """01 문서 5-1 표의 80°C 60분 DDI 증가 예측을 모두 계산한다 (cycle).

    openwetware는 표의 원 조건대로 Ea 121로 환산하고, 나머지는 Ea_kJ(기본 130)를 쓴다.
    """
    return {
        "openwetware": ddi_increase(openwetware_rate_per_min(), OPENWETWARE_T_C,
                                    OPENWETWARE_EA_KJ, efficiency),
        "hird_boil_max": ddi_increase(hird_boil_rate_per_min(efficiency), HIRD_BOIL_T_C,
                                      Ea_kJ, efficiency),
        "hird_autoclave": ddi_increase(hird_autoclave_rate_per_min(efficiency), HIRD_AUTO_T_C,
                                       Ea_kJ, efficiency),
        "brisco": ddi_increase(brisco_rate_per_min(), BRISCO_T_C, Ea_kJ, efficiency),
    }
