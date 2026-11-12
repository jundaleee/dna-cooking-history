# DNA 손상으로 조리 이력을 읽는 연구를 위한 절차와 프로토콜

익힌 고기의 DNA 손상 지표(두 구간의 qPCR Ct 차이, DDI)로 조리 온도·시간을 추정하려는 **후속 연구**가 그대로 가져다 쓸 수 있도록, 실험 전 계산, 실험 프로토콜, 측정 점검, 평가 방법과 그 도구를 모았다.

대구과학고등학교 정보 R&E(2026) 「머신러닝 기반 DNA 붕괴 패턴을 활용한 식품 가공 이력 추적 모델 개발 연구」에서 세우고 시험한 절차이다. 이 저장소에는 그 연구의 결과와 원자료를 담지 않았다.

## 무엇이 있나

| 경로 | 내용 |
| --- | --- |
| [`PROCEDURE.md`](PROCEDURE.md) | 검증 절차 10단계: 정보량 상한 계산부터 독립 재계산까지, 단계별 통과 기준과 도구 |
| [`protocols/01_sample_and_heating.md`](protocols/01_sample_and_heating.md) | 시료 준비, 수비드 가열, 탐침 온도 측정 |
| [`protocols/02_extraction_and_qc.md`](protocols/02_extraction_and_qc.md) | 추출 배치 원칙과 품질 기록 |
| [`protocols/03_qpcr_plate.md`](protocols/03_qpcr_plate.md) | 프라이머, 반응 조성, 프로그램, 판 배치, 판 돌리기 전 점검표 |
| [`protocols/04_pilot_plate.md`](protocols/04_pilot_plate.md) | 19칸 예비 판과 판정 규칙 R1–R7 |
| [`protocols/05_preregistration.md`](protocols/05_preregistration.md) | 사전 고정, 원자료 수령 기록, 독립 재계산 |
| [`protocols/06_analysis.md`](protocols/06_analysis.md) | 원신호 Ct, 대조 칸 판정, 정상 증폭 점검, 평가 단위, 정보량 보고 |
| [`templates/`](templates/) | 판 배치 예시, 표준곡선 배치, 시료표, 표적 이름 대응표, 원자료 수령 기록 양식 |
| [`src/rne/`](src/rne/) | 도구(아래) |

## 도구

| 단계 | 명령 | 하는 일 |
| --- | --- | --- |
| 실험 전 | `python -m rne.design --temps 50,60,70,75,80 --times 15,30,60 --reps 3,2,3 --D60 0.9 --pass-at 75,15 --power --budget` | 측정이 조건에 대해 가질 수 있는 최대 정보(비트), 오류율 하한(파노), 합격·불합격 최소 오류율, 검정력, 비트 단위 오차 예산 |
| 판 설계 | `python -m rne.design.layout --temps ... --times ... --reps ... --out plate_layout.csv` | 두 구간 이웃 칸, 무작위 위치, 대조 칸 16개의 96칸 배치표 |
| 사전 고정 | `python -m rne.pipeline.freeze make --settings settings.json --out manifest.json` | 코드·설정·입력 파일의 해시값 기록과 확인 |
| 측정 점검 | `python -m rne.qpcr.raw_ct --export plate.xls` | 기준 염료로 나누기 전 원신호로 Ct 다시 내기 |
| | `python -m rne.qpcr.controls --export plate.xls --layout plate_layout.csv --out out/` | 재료를 뺀 대조 칸 판정 |
| | `python -m rne.qc.normal_range --xls plate.xls --layout plate_layout.csv --out out/` | 정상 증폭 범위 점검(가상 정상 곡선으로 학습한 분포 밖 탐지) |
| 평가 | `rne.evaluate.compare_schemes(model, X, y, condition, batch)` | 시료 하나씩·조건 단위·배치 단위 평가와 배치 안 순열 p |

입력은 StepOne/StepOnePlus 내보내기(.xls: Multicomponent, Amplification, Melt 시트)와 판 배치표(CSV: `well, target, kind`)이다. 실험 전 계산은 손상 모형 안의 가상 수치이며 실측 결과가 아니다.

## 설치

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

Python 3.10 이상. 시험한 환경: Python 3.14, numpy 2.5, pandas 3.1, scikit-learn 1.9.
