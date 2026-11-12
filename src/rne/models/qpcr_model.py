"""qPCR 측정 모델 (mtDNA).

"""
from __future__ import annotations

import numpy as np

from rne.common.schema import DELTA_BP
from rne.models.kinetics import ArrayLike, _out


def ct_from_lambda(lam_per_nt: ArrayLike, length_bp: float, ct0: float,
                   template_ratio: ArrayLike = 1.0, efficiency: float = 1.0) -> ArrayLike:
    """손상 밀도에서 Ct를 계산한다 (cycle).

    Ct_L = Ct₀,L + (λ·L − ln(N/N₀)) / ln(1 + E_L).

    Args:
        lam_per_nt: nt당 손상 수 λ.
        length_bp: 앰플리콘 길이 (bp).
        ct0: 손상 0, N = N₀일 때의 Ct.
        template_ratio: 넣은 주형 양의 비 N/N₀ (무차원).
        efficiency: 증폭 효율 E (1.0 = 100%).
    """
    lam = np.asarray(lam_per_nt, dtype=float)
    ratio = np.asarray(template_ratio, dtype=float)
    return _out(ct0 + (lam * length_bp - np.log(ratio)) / np.log1p(efficiency))


def ddi_from_lambda(lam_mt: ArrayLike, ddi0: float = 0.0, efficiency: float = 1.0) -> ArrayLike:
    """mt 손상 밀도에서 DDI = Ct₅₈₆ − Ct₁₀₄를 계산한다 (cycle).

    DDI = DDI₀ + 482·λ_mt / ln(1 + E). 두 표적의 E가 같을 때. 482 = schema.DELTA_BP.
    """
    lam = np.asarray(lam_mt, dtype=float)
    return _out(ddi0 + DELTA_BP * lam / np.log1p(efficiency))


def lambda_from_ddi(ddi: ArrayLike, ddi0: float = 0.0, efficiency: float = 1.0) -> ArrayLike:
    """DDI에서 mt 손상 밀도 λ를 되돌린다 (nt⁻¹). ddi_from_lambda의 역함수.

    """
    d = np.asarray(ddi, dtype=float)
    return _out((d - ddi0) * np.log1p(efficiency) / DELTA_BP)
