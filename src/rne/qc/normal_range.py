"""정상 증폭 범위 점검: PCR의 물리적 제약을 지키는 가상 정상 곡선으로 학습한 분포 밖 탐지기.

정상 증폭이 지켜야 하는 조건(cycle당 증가 배율 ≤ 1+E ≤ 2, 기준 염료(ROX) 일정, 주형·프라이머가 없으면 평평,
산물이 있으면 멜트에서 녹는 봉우리)을 넓은 모수 범위에서 만족하는 가상 곡선을 만들고, 그 특징 분포로
마할라노비스 거리 탐지기를 학습한 뒤 분할 등각(split conformal) 문턱으로 실측 칸이 정상 곡선의 분포 밖인지 판정한다.
시료 칸은 'amp' 탐지기(배율·기준 염료·멜트), 증폭이 불가능한 대조 칸(주형 없음, 프라이머 없음, 믹스만)은 'neg'
탐지기(상승·기준 염료)로 본다. 물만 넣은 칸은 염료가 없어 뺀다.

입력: StepOne/StepOnePlus 내보내기(.xls, Multicomponent·Melt 시트)와 판 배치표(CSV: well, kind 열;
kind ∈ sample, ntc, noprimer, mixonly, water). 'Sample Name' 열은 읽지 않는다.
실행: python -m rne.qc.normal_range --xls <판 .xls> --layout <배치표.csv> --out <폴더> [--aggregate-only]
가상 곡선의 성능 수치는 탐지기 점검이며 실측 결과가 아니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

SEED = 20261111
ALPHA = 0.05            # 등각 문턱의 명목 오거부율
N_CYCLES = 40
BASE_CYC = slice(1, 5)  # cycle 2–5 (0부터 셈)
ROX_START = slice(1, 4)  # cycle 2–4
ROX_END = slice(37, 40)  # cycle 38–40
MELT_RANGE_C = (65.0, 92.0)


# --------------------------------------------------------------------------- 읽기
def _rows(sheet, start=8):
    """헤더(7행) 아래 자료 행을 돌려준다."""
    for r in range(start, sheet.nrows):
        yield sheet.row_values(r)


def read_plate(xls_path: str) -> dict:
    """StepOne .xls에서 칸별 SYBR·ROX(cycle 1–40)와 멜트(온도, 형광)를 읽는다. 시료 이름은 읽지 않는다."""
    wb = xlrd.open_workbook(xls_path)
    target = {}
    for row in _rows(wb.sheet_by_name("Sample Setup")):
        if row[0]:
            target[str(row[0])] = str(row[5])          # 5 = Target Name (1 = Sample Name은 읽지 않음)
    sybr = {w: np.zeros(N_CYCLES) for w in target}
    rox = {w: np.zeros(N_CYCLES) for w in target}
    for row in _rows(wb.sheet_by_name("Multicomponent Data")):
        w, c = str(row[0]), int(row[1])
        if w in sybr and 1 <= c <= N_CYCLES:
            sybr[w][c - 1], rox[w][c - 1] = float(row[2]), float(row[3])
    temp, fluo = {}, {}
    for name, store in (("Melt Region Temperature Data", temp), ("Melt Region Normalized Data", fluo)):
        for row in _rows(wb.sheet_by_name(name)):
            store[str(row[1])] = np.array([float(v) for v in row[4:] if v != ""])
    return {"target": target, "sybr": sybr, "rox": rox, "melt_T": temp, "melt_F": fluo}


# --------------------------------------------------------------------------- 특징
def melt_prominence(T_C: np.ndarray, F: np.ndarray) -> float:
    """멜트 봉우리 뚜렷함: −dF/dT(5점 이동평균)의 최댓값 ÷ 중앙값, 65–92 °C. 산물이 녹으면 커진다."""
    d = -np.gradient(F, T_C)
    d = np.convolve(d, np.ones(5) / 5, mode="same")
    m = (T_C >= MELT_RANGE_C[0]) & (T_C <= MELT_RANGE_C[1])
    med = np.median(np.abs(d[m]))
    return float(np.max(d[m]) / med) if med > 0 else np.nan


def curve_features(sybr: np.ndarray, rox: np.ndarray, T_C: np.ndarray, F: np.ndarray) -> dict:
    """칸 하나의 크기 없는 특징.

    - rise: (끝 3 cycle 평균 − 기준선) ÷ 기준선. 기준선은 cycle 2–5 중앙값
    - g_max: 증폭 초기 구간(ΔF가 최대의 5–50 %)에서 cycle당 배율 ΔF_{c+1}/ΔF_c의 최댓값. 정상은 ≤ 1+E ≤ 2
    - rox_fold: ROX cycle 38–40 평균 ÷ cycle 2–4 평균. 정상은 ≈ 1
    - melt_prom: melt_prominence
    """
    base = np.median(sybr[BASE_CYC])
    dF = sybr - base
    top = dF.max()
    rise = float(np.mean(dF[-3:]) / base)
    g = np.nan
    if top > 0:
        idx = [c for c in range(N_CYCLES - 1)
               if 0.05 * top < dF[c] < 0.5 * top and dF[c + 1] > 0]
        if idx:
            g = float(max(dF[c + 1] / dF[c] for c in idx))
    return {"rise": rise, "g_max": g,
            "rox_fold": float(np.mean(rox[ROX_END]) / np.mean(rox[ROX_START])),
            "melt_prom": melt_prominence(T_C, F)}


# --------------------------------------------------------------------------- 가상 정상 곡선
def simulate_normal(n: int, kind: str, T_grid: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """정상 qPCR 칸의 특징을 n개 만든다. kind = 'amp'(주형+프라이머) 또는 'neg'(증폭 불가 칸).

    가정(넓게 잡음, MIQE의 효율 90–110 %보다 넓힘):
    - 산물 N_c = N0·(1+E)^c, E ~ U(0.80, 1.00); 형광 ΔF_c = A·N_c/(N_c + K)이라 cycle당 배율 ≤ 1+E ≤ 2
    - Ct(반포화 cycle) ~ U(8, 35), 고원 A/기준선 ~ U(0.5, 40)
    - 기준선 표류 ±0.3 %/cycle, 측정 잡음 0.3–1 % (곱셈)
    - ROX: 처음 값 × (1 + 표류), 전체 표류 ~ U(−0.15, 0.15), SYBR→ROX 새어 듦 b ~ U(0, 0.02), 잡음 0.5 %
    - 멜트: 바탕 형광이 온도에 따라 1 °C당 0.5–2 % 줄고, 산물(몫 0.3–0.9)이 Tm ~ U(74, 88), 폭 0.4–1.5 °C로 녹음
    - 증폭 불가 칸: 산물 없음. NTC형 늦은 프라이머 이합체(확률 0.3, Ct 34–40, 고원 0.05–0.5, Tm 70–80)
    """
    out = []
    cyc = np.arange(1, N_CYCLES + 1)
    for _ in range(n):
        base = rng.uniform(2e4, 1e5)
        noise = rng.uniform(0.003, 0.01)
        drift = rng.uniform(-0.003, 0.003) * (cyc - 1)
        dF = np.zeros(N_CYCLES)
        p_prod, Tm, width = 0.0, 80.0, 1.0
        amp = kind == "amp" or rng.random() < 0.3
        if amp:
            E = rng.uniform(0.80, 1.00)
            ct = rng.uniform(8, 35) if kind == "amp" else rng.uniform(34, 40)
            A = base * (rng.uniform(0.5, 40) if kind == "amp" else rng.uniform(0.05, 0.5))
            K = 1.0
            N = K * (1 + E) ** (cyc - ct)          # N = K at the half-saturation cycle
            dF = A * N / (N + K)
            p_prod = rng.uniform(0.3, 0.9) if kind == "amp" else rng.uniform(0.1, 0.4)
            Tm = rng.uniform(74, 88) if kind == "amp" else rng.uniform(70, 80)
            width = rng.uniform(0.4, 1.5)
        sybr = (base * (1 + drift) + dF) * (1 + rng.normal(0, noise, N_CYCLES))
        rox0 = rng.uniform(2e4, 6e4)
        rox_drift = rng.uniform(-0.15, 0.15) * (cyc - 1) / (N_CYCLES - 1)
        rox = (rox0 * (1 + rox_drift) + rng.uniform(0, 0.02) * (sybr - base)) \
            * (1 + rng.normal(0, 0.005, N_CYCLES))
        k = rng.uniform(0.005, 0.02)
        bg = np.clip(1 - k * (T_grid - T_grid[0]), 0.05, None)
        prod = 1 / (1 + np.exp((T_grid - Tm) / width))
        F = (1 - p_prod) * bg + p_prod * prod
        F = F * (1 + rng.normal(0, 0.002, T_grid.size))
        out.append(curve_features(sybr, rox, T_grid, F))
    return out


# --------------------------------------------------------------------------- 탐지기
FEATS = {"amp": ("g_max", "log_rox_fold", "log_melt_prom"),
         "neg": ("log_rise", "log_rox_fold")}


def to_matrix(rows: list[dict], kind: str) -> np.ndarray:
    """특징 행렬. 로그는 자연로그, rise는 log(1+rise)."""
    cols = []
    for name in FEATS[kind]:
        if name == "g_max":
            cols.append([r["g_max"] for r in rows])
        elif name == "log_rox_fold":
            cols.append([np.log(r["rox_fold"]) for r in rows])
        elif name == "log_melt_prom":
            cols.append([np.log(r["melt_prom"]) for r in rows])
        elif name == "log_rise":
            cols.append([np.log1p(max(r["rise"], -0.99)) for r in rows])
    return np.array(cols, dtype=float).T


class MahalanobisOOD:
    """d(x) = √((x−μ)ᵀ Σ⁻¹ (x−μ)), 문턱 τ는 분할 등각: 보정 점수의 ⌈(n+1)(1−α)⌉번째 값."""

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray, alpha: float = ALPHA):
        X_train = X_train[np.all(np.isfinite(X_train), axis=1)]
        self.mu = X_train.mean(axis=0)
        self.P = np.linalg.pinv(np.cov(X_train, rowvar=False))
        s = np.sort(self.score(X_cal))
        k = int(np.ceil((s.size + 1) * (1 - alpha)))
        self.tau = float(s[min(k, s.size) - 1])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """특징이 정의되지 않은 칸(NaN: 증폭 구간 없음)은 점수 무한대 = 분포 밖."""
        D = X - self.mu
        d = np.sqrt(np.einsum("ij,jk,ik->i", D, self.P, D))
        return np.where(np.all(np.isfinite(X), axis=1), d, np.inf)


# --------------------------------------------------------------------------- 점검(가상)
def inject(rows: list[dict], what: str) -> list[dict]:
    """정상 가상 칸에 실측에서 본 이상 하나를 넣는다(검출력 점검용)."""
    out = []
    for r in rows:
        r = dict(r)
        if what == "rox_tracks":         # 2차 판처럼 ROX가 5배 오름
            r["rox_fold"] = 5.0
        elif what == "no_melt":          # 산물 봉우리 없음(실측 시료 중앙 1.14)
            r["melt_prom"] = 1.15
        elif what == "fast_growth":      # cycle당 2.4배
            r["g_max"] = 2.4
        elif what == "neg_rises":        # 증폭 불가 칸이 20배 오름
            r["rise"] = 20.0
        out.append(r)
    return out



NEGATIVE_KINDS = ("ntc", "noprimer", "mixonly")


def load_kinds(layout_csv) -> dict:
    """배치표에서 칸 종류를 읽는다(well, kind 열)."""
    d = pd.read_csv(layout_csv, encoding="utf-8-sig")
    return {str(w): str(k) for w, k in zip(d["well"], d["kind"])}


def fit_detectors(T_grid: np.ndarray, n: int = 4000, seed: int = SEED) -> tuple[dict, dict]:
    """가상 정상 곡선으로 'amp'·'neg' 탐지기를 학습·보정하고, 따로 만든 시험 곡선으로 오거부율과 이상 검출률을 잰다."""
    det, check = {}, {}
    for kind in ("amp", "neg"):
        tr, ca, te = (simulate_normal(n, kind, T_grid, np.random.default_rng([seed, i, int(kind == "amp")]))
                      for i in range(3))
        det[kind] = MahalanobisOOD().fit(to_matrix(tr, kind), to_matrix(ca, kind))
        fpr = float(np.mean(det[kind].score(to_matrix(te, kind)) > det[kind].tau))
        injected = {}
        for what in (("rox_tracks", "no_melt", "fast_growth") if kind == "amp" else ("neg_rises", "rox_tracks")):
            injected[what] = float(np.mean(det[kind].score(to_matrix(inject(te, what), kind)) > det[kind].tau))
        check[kind] = {"tau": det[kind].tau, "false_reject_rate_test": fpr, "detect_rate_injected": injected, "n": n}
    return det, check


def check_plate(xls, layout_csv, n: int = 4000, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """칸마다 분포 밖 점수와 판정. 반환: 칸별 표(well, kind, detector, score, tau, outside)와 가상 점검 요약."""
    plate = read_plate(str(xls))
    kinds = load_kinds(layout_csv)
    T_grid = next(iter(plate["melt_T"].values()))
    det, check = fit_detectors(T_grid, n, seed)
    rows = []
    for w in plate["target"]:
        k = kinds.get(w)
        if k == "sample":
            kind = "amp"
        elif k in NEGATIVE_KINDS:
            kind = "neg"
        else:
            continue
        f = curve_features(plate["sybr"][w], plate["rox"][w], plate["melt_T"][w], plate["melt_F"][w])
        s = float(det[kind].score(to_matrix([f], kind))[0])
        rows.append({"well": w, "kind": k, "detector": kind, "score": s, "tau": det[kind].tau,
                     "outside": bool(s > det[kind].tau), **f})
    return pd.DataFrame(rows), check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="정상 증폭 범위 점검(분포 밖 탐지)")
    ap.add_argument("--xls", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", default="outputs/normal_range")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--aggregate-only", action="store_true", help="칸별 값을 쓰지 않고 종류별 집계만(눈가림 유지)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, check = check_plate(a.xls, a.layout, a.n)
    summary = {"input_sha256": hashlib.sha256(Path(a.xls).read_bytes()).hexdigest(), "seed": SEED, "alpha": ALPHA,
               "synthetic_check": check,
               "by_kind": {k: {"n": int(len(g)), "frac_outside": float(g.outside.mean())} for k, g in table.groupby("kind")}}
    (out / "normal_range_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if not a.aggregate_only:
        table.to_csv(out / "normal_range_wells.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
