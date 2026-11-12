"""실험 전 계산을 한 번에: 정보량, 오류율 하한, (선택) 합격·불합격 최소 오류율, 검정력, 오차 예산.

    python -m rne.design --temps 50,60,70,75,80 --times 15,30,60 --reps 3,2,3 --D60 0.9 --sigma 0.42 \
        --pass-at 75,15 --power --budget --batch-temps 60,70
"""
from __future__ import annotations

import argparse
import json

from rne.design.error_budget import error_budget
from rne.design.information import design_information
from rne.design.power import detection_power


def _nums(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="실험 전 정보량·오류율·검정력 계산(가상 수치, 설계 점검용)")
    ap.add_argument("--temps", required=True, help="가열 온도(°C), 쉼표로")
    ap.add_argument("--times", required=True, help="가열 시간(분), 쉼표로")
    ap.add_argument("--reps", required=True, help="반복 수: 정수, 시간마다, 또는 조건마다(쉼표로)")
    ap.add_argument("--D60", type=float, required=True, help="80 °C 60분의 DDI 변화(cycle)")
    ap.add_argument("--sigma", type=float, default=0.42, help="시료 하나의 DDI 잡음 SD(cycle)")
    ap.add_argument("--pass-at", default=None, help="합격 기준 가열 '온도,시간'(이 가열량 이상을 합격)")
    ap.add_argument("--power", action="store_true", help="검정력도 계산(수 분)")
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--budget", action="store_true", help="비트 단위 오차 예산도 계산")
    ap.add_argument("--batch-temps", default="", help="다른 추출 배치에서 나온 온도(오차 예산용)")
    a = ap.parse_args(argv)
    reps = [int(x) for x in _nums(a.reps)]
    reps = reps[0] if len(reps) == 1 else reps
    pass_at = _nums(a.pass_at) if a.pass_at else None
    res = design_information(_nums(a.temps), _nums(a.times), reps, a.D60, a.sigma, pass_at=pass_at)
    if a.power:
        res["detection_power"] = detection_power(_nums(a.temps), _nums(a.times), reps, a.D60, a.sigma,
                                                 n_sim=a.n_sim, n_jobs=-1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.budget:
        print(error_budget(_nums(a.temps), _nums(a.times), reps, a.D60, a.sigma,
                           batch_temps=_nums(a.batch_temps)).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
