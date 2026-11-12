"""기기 판단을 믿지 않고 원시 곡선에서 다시 계산한다: Ct, 웰별 효율, ROX 진단, 멜트 요약.

1차 판(11/4)은 자동 기준선이 웰마다 달랐고(A1·B1 3–34, C1–F1 1–2 …) 전 웰 BADROX였다
. 여기서는 모든 웰에 같은 수동 기준선과 표적별 하나의 문턱을 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 웰별 증폭 배수(1+E) 허용 범위. 밖이면 플래그
AMP_FACTOR_RANGE = (1.6, 2.1)


# --------------------------------------------------------------------------- 기준선·Ct


def subtract_baseline_adaptive(amplification: pd.DataFrame, baseline_cycles=(1, 3)) -> pd.DataFrame:
    """적응 기준선: 웰마다 기준선 끝 = 처음으로 Rn − Rn(첫 cycle)이 판 잡음의 5√2배를 넘는 cycle − 1
    (최소 baseline_cycles 시작, 최대 끝). 기준선은 기울기 없는 평균이다. 기준선 1–3에서 이미 오르는 이른 웰이
    기울기 외삽 때문에 Ct를 잃던 문제를 막는다. 반환 열은 subtract_baseline과 같다.
    """
    lo, hi = baseline_cycles
    a = amplification.sort_values(["well", "cycle"])
    w = a.pivot(index="well", columns="cycle", values="rn")
    noise = float(np.nanmedian(np.abs(np.diff(w.to_numpy(float)[:, :max(hi, 3)], 2, axis=1)))) / np.sqrt(6) + 1e-12
    parts = []
    for well, g in a.groupby("well", sort=False):
        g = g.copy()
        c = g["cycle"].to_numpy(float)
        rn = g["rn"].to_numpy(float)
        k = np.nonzero((rn - rn[0]) > 5 * np.sqrt(2) * noise)[0]
        end = int(np.clip((c[k[0]] - 1) if len(k) else hi, lo, hi))
        sel = (c >= lo) & (c <= end) & np.isfinite(rn)
        b = float(np.nanmean(rn[sel])) if sel.any() else float(rn[0])
        g["drn"] = rn - b
        g["baseline_sd"] = noise
        g["baseline_slope"] = 0.0
        g["baseline_end_adaptive"] = end
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def subtract_baseline(amplification: pd.DataFrame, baseline_cycles=(3, 15)) -> pd.DataFrame:
    """모든 웰에 같은 사이클 구간으로 직선 기준선을 맞춰 뺀다.

    입력: well, target, cycle, rn. 반환: 입력 + `drn`(Rn − 기준선), 웰별 `baseline_sd`(구간 잔차 SD),
    `baseline_slope`(Rn/cycle). 구간은 양 끝 포함.
    """
    lo, hi = baseline_cycles
    parts = []
    for well, g in amplification.groupby("well", sort=False):
        g = g.sort_values("cycle").copy()
        c = g["cycle"].to_numpy(float)
        rn = g["rn"].to_numpy(float)
        sel = (c >= lo) & (c <= hi) & np.isfinite(rn)
        if sel.sum() >= 2:
            coef = np.polyfit(c[sel], rn[sel], 1)
            resid = rn[sel] - np.polyval(coef, c[sel])
            sd = float(np.std(resid, ddof=2)) if sel.sum() > 2 else 0.0
        else:
            coef, sd = np.array([0.0, np.nanmean(rn[:3]) if len(rn) else 0.0]), np.nan
        g["drn"] = rn - np.polyval(coef, c)
        g["baseline_sd"] = sd
        g["baseline_slope"] = float(coef[0])
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else amplification.assign(drn=[], baseline_sd=[], baseline_slope=[])


def auto_threshold(drn_table: pd.DataFrame, noise_mult: float = 10.0, top_frac: float = 0.10) -> float:
    """표적 하나의 문턱: 지수 구간 안. 하한 = 웰들 기준선 SD의 중앙값 × noise_mult,
    상한 = 증폭 웰 정체기 ΔRn의 최소 × top_frac. 둘의 기하 평균(하한 ≥ 상한이면 하한).

    하한은 최소 정체기 중앙값의 2 %(잡음이 0이어도 기준선 끝자락 증폭에 휘둘리지 않게).
    drn_table은 subtract_baseline 결과(한 표적).
    """
    per = drn_table.groupby("well").agg(plateau=("drn", "max"), sd=("baseline_sd", "first"))
    # 흔들리는 웰 하나(ROX 이상 등)에 문턱이 끌려 올라가지 않도록 기준선 SD는 중앙값을 쓴다
    sd_max = float(np.nanmedian(per["sd"])) if per["sd"].notna().any() else 0.0
    amp = per[per["plateau"] > max(20 * sd_max, 1e-9)]
    if amp.empty:
        return float("nan")
    plateau_med = float(amp["plateau"].median())
    lower = max(noise_mult * sd_max, 0.02 * plateau_med)
    # 정체기가 매우 낮은 웰(비특이·실패)에 끌려가지 않게 중앙값의 절반 이상인 웰만으로 상한을 잡는다
    upper = top_frac * float(amp.loc[amp["plateau"] >= 0.5 * plateau_med, "plateau"].min())
    return float(np.sqrt(lower * upper)) if upper > lower else float(lower)


def _crossing(c: np.ndarray, y: np.ndarray, thr: float) -> float:
    """y가 thr를 처음 넘는 사이클. 로그 선형 보간(지수 구간이라 로그에서 직선). 없으면 NaN.

    잡음에 의한 이른 교차를 피하려고, 교차 뒤로 끝까지 thr 밑으로 다시 내려가지 않는 마지막 교차를 쓴다.
    """
    above = y >= thr
    if not above.any():
        return float("nan")
    # 마지막으로 아래였던 위치 다음이 교차
    below_idx = np.where(~above)[0]
    if len(below_idx) == 0:
        return float(c[0])
    i = below_idx[-1]
    if i + 1 >= len(c):
        return float("nan")
    y0, y1 = y[i], y[i + 1]
    if y0 > 0 and y1 > 0:
        f = (np.log(thr) - np.log(y0)) / (np.log(y1) - np.log(y0))
    else:
        f = (thr - y0) / (y1 - y0)
    return float(c[i] + f * (c[i + 1] - c[i]))


def well_exponential_window(drn_table: pd.DataFrame, noise_mult: float = 5.0, upper_frac_of_sdm: float = 0.5,
                            min_rise: float = 20.0) -> pd.DataFrame:
    """웰마다 지수 구간의 ΔRn 범위 [lower, upper]를 곡선 모양만으로 정한다.

    - lower = noise_mult × 판 잡음의 중앙값 (잡음 위). 웰 잡음 = max(기준선 잔차 SD, 2차 차분 MAD/√6). 웰별 SD가 아니라 판 중앙값을 쓴다: Ct가 이른 웰은
      증폭 시작이 기준선 구간에 섞여 SD가 부풀고, 흔들리는 웰 하나가 하한을 끌어올리지 않게 하려는 것
    - upper = upper_frac_of_sdm × (2차 미분 최대 사이클의 ΔRn). 2차 미분 최대(SDM)는 지수 구간이 끝나고
      곡선이 휘기 시작하는 곳이다. 로지스틱이면 SDM에서 ΔRn ≈ 고원의 21 %이므로 upper ≈ 고원의 10 %
    - 증폭 안 한 웰(최대 ΔRn < min_rise × SD, 또는 SD가 0이면 최대 ΔRn ≤ 0): amplified=False
    시료의 조건·라벨·주형량은 쓰지 않는다. 입력은 subtract_baseline 결과(well, cycle, drn, baseline_sd).
    반환 열: well, amplified, sdm_cycle, drn_at_sdm, lower, upper, plateau_drn
    """
    rows = []
    # 잡음: 기준선 잔차 SD와 곡선 전체의 고주파 잡음(2차 차분의 MAD/√6) 중 큰 값, 판 중앙값.
    # 기준선이 3 cycle이면 직선 잔차 SD는 자유도 1이라 잡음을 크게 낮춰 잡는다(2차 판 실측에서 확인)
    def _hf(g):
        y = g.sort_values("cycle")["drn"].to_numpy(float)
        return float(1.4826 * np.median(np.abs(np.diff(y, 2))) / np.sqrt(6.0)) if len(y) >= 6 else 0.0
    if len(drn_table):
        sd_w = drn_table.groupby("well")["baseline_sd"].first().fillna(0.0)
        hf_w = drn_table.groupby("well").apply(_hf, include_groups=False)
        sd_med = float(np.nanmedian(np.maximum(sd_w.to_numpy(float), hf_w.reindex(sd_w.index).to_numpy(float))))
    else:
        sd_med = 0.0
    sd_med = sd_med if np.isfinite(sd_med) else 0.0
    for well, g in drn_table.groupby("well", sort=False):
        g = g.sort_values("cycle")
        c = g["cycle"].to_numpy(float)
        y = g["drn"].to_numpy(float)
        sd = float(g["baseline_sd"].iloc[0]) if np.isfinite(g["baseline_sd"].iloc[0]) else 0.0
        top = float(np.nanmax(y)) if len(y) else np.nan
        amplified = bool(np.isfinite(top) and top > max(min_rise * sd, 0.0) and top > 0)
        sdm_c = drn_sdm = lower = upper = np.nan
        if amplified and len(y) >= 5:
            ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().to_numpy()
            d2 = ys[2:] - 2 * ys[1:-1] + ys[:-2]
            k = int(np.nanargmax(d2)) + 1
            sdm_c, drn_sdm = float(c[k]), float(y[k])
            lower = max(noise_mult * sd_med, 1e-6 * top)
            upper = upper_frac_of_sdm * drn_sdm
        rows.append(dict(well=well, amplified=amplified, sdm_cycle=sdm_c, drn_at_sdm=drn_sdm,
                         lower=lower, upper=upper, plateau_drn=top))
    return pd.DataFrame(rows, columns=["well", "amplified", "sdm_cycle", "drn_at_sdm", "lower", "upper", "plateau_drn"])


def exponential_phase_threshold(drn_table: pd.DataFrame, **kw) -> tuple[float, dict]:
    """표적 하나의 문턱을 모든 증폭 웰의 지수 구간 안에 들도록 정한다(곡선 모양만 씀).

    공통 창 = [증폭 웰 lower의 최대, 증폭 웰 upper의 최소] (lower는 판 기준선 SD 중앙값으로 정함).
    창이 있으면 그 기하 평균.
    창이 없으면(웰 고원 차이가 너무 크거나 잡음이 큼) 웰별 창 기하 평균의 중앙값을 쓰고 `no_common_window`.
    반환: (문턱, 정보 dict: n_amplified, window_lo, window_hi, rule)
    """
    w = well_exponential_window(drn_table, **kw)
    a = w[w["amplified"] & w["upper"].notna() & (w["upper"] > w["lower"])]
    if a.empty:
        return float("nan"), dict(n_amplified=0, window_lo=np.nan, window_hi=np.nan, rule="no_amplified_well")
    lo_c, hi_c = float(a["lower"].max()), float(a["upper"].min())
    if hi_c > lo_c:
        return float(np.sqrt(lo_c * hi_c)), dict(n_amplified=int(len(a)), window_lo=lo_c, window_hi=hi_c, rule="common_window")
    per = np.sqrt(a["lower"] * a["upper"])
    return float(per.median()), dict(n_amplified=int(len(a)), window_lo=lo_c, window_hi=hi_c, rule="no_common_window")


def threshold_position(drn_table: pd.DataFrame, threshold: float, **kw) -> pd.DataFrame:
    """주어진 문턱이 웰마다 곡선의 어디에 있는지: `exponential` / `above_exponential`(휘는 곳·고원) /
    `below_noise` / `not_reached`(고원이 문턱보다 낮음) / `not_amplified`.

    2차 판 사진에서 문턱 1.5가 고원(약 0.4–3) 안에 있던 문제를 잡는 점검이다.
    """
    w = well_exponential_window(drn_table, **kw)
    pos = []
    for r in w.itertuples(index=False):
        if not r.amplified:
            pos.append("not_amplified")
        elif threshold >= r.plateau_drn:
            pos.append("not_reached")
        elif np.isfinite(r.upper) and threshold > r.upper:
            pos.append("above_exponential")
        elif np.isfinite(r.lower) and threshold < r.lower:
            pos.append("below_noise")
        else:
            pos.append("exponential")
    w["threshold"] = threshold
    w["position"] = pos
    return w


def recompute_ct(amplification: pd.DataFrame, baseline_cycles=(3, 15), threshold=None,
                 instrument: pd.DataFrame | None = None, no_amp_frac: float | None = 0.2,
                 baseline_mode: str = "fixed") -> pd.DataFrame:
    """모든 웰에 같은 수동 기준선을 쓰고, 표적별 문턱 하나로 Ct를 다시 계산한다.

    - `threshold`: None이면 표적마다 auto_threshold, `"exponential"`이면 exponential_phase_threshold
     , float면 모든 표적, dict면 {표적: 문턱}(ΔRn 단위)
    - `instrument`: qpcr_long(read_steponeplus 결과)을 주면 기기 Ct를 붙이고 `ct_diff = ct_recomputed − ct_instrument`
    반환 열: well, target, baseline_start, baseline_end, threshold, ct_recomputed, undetermined_recomputed,
    plateau_drn, baseline_sd, baseline_slope, threshold_position, recompute_flags (+ ct_instrument, ct_diff)
    플래그: `early_amp`(Ct가 기준선 끝 + 5 사이클 안 → 기준선 구간이 증폭을 먹었을 수 있음), `no_threshold`,
    `threshold_not_exponential`(이 웰에서 문턱이 지수 구간 밖), `no_amplification`(최대 ΔRn이 판 고원 중앙값의
    no_amp_frac배 또는 기준선 잡음 SD의 10배에 못 미침 → Ct 없음. None이면 끈다)
    """
    # baseline_mode: 'fixed'(모든 웰 같은 구간 직선, 지금 기본) 또는 'adaptive'
    sub = (subtract_baseline_adaptive(amplification, baseline_cycles) if baseline_mode == "adaptive"
           else subtract_baseline(amplification, baseline_cycles))
    lo, hi = baseline_cycles
    rows = []
    # 판 전체(모든 표적) 고원 중앙값: 표적 이름이 갈라져 한 무리에 평평한 웰만 남으면 무리 안 중앙값이 잡음 크기가 된다
    # (가짜 판 스트레스: 표적 이름 변형에서 NTC 평평한 곡선이 Ct 10.0을 받아 0단계 실패). 두 기준 중 큰 쪽을 쓴다
    # 증폭 없음 판정은 기울기 없는 기준선(1–3 평균)으로 잰 상승량을 쓴다: 기준선 세 점의 잡음 기울기를 40 cycle로
    # 늘이면 평평한 NTC도 ΔRn 0.3까지 '오른다'(가짜 판 스트레스 시험, D8). 원신호 rn이 없으면 drn을 쓴다
    if "rn" in amplification:
        a = amplification.sort_values(["well", "cycle"])
        b0 = a[(a["cycle"] >= lo) & (a["cycle"] <= hi)].groupby("well")["rn"].mean()
        rise = a.groupby("well")["rn"].max() - b0
    else:
        rise = sub.groupby("well")["drn"].max()
    rise_t = {t: rise.reindex(g["well"].unique()) for t, g in sub.groupby("target", sort=False)}
    plate_med = float(rise.median()) if len(rise) else np.nan
    for target, gt in sub.groupby("target", sort=False):
        if isinstance(threshold, dict):
            thr = float(threshold.get(target, np.nan))
        elif isinstance(threshold, str) and threshold == "exponential":
            thr, _ = exponential_phase_threshold(gt)
        elif isinstance(threshold, str) and threshold.startswith("plateau_"):
            # 판(표적) 고원 중앙값의 일정 비율. 곡선 모양만 쓴다
            thr = float(threshold.split("_", 1)[1]) * float(gt.groupby("well")["drn"].max().median())
        elif threshold is not None:
            thr = float(threshold)
        else:
            thr = auto_threshold(gt)
        posmap = (threshold_position(gt, thr).set_index("well")["position"].to_dict()
                  if np.isfinite(thr) else {})
        # 증폭 없음 판정: 평평한 곡선(실패 웰, 깨끗한 NTC)은 낮은 문턱에서 잡음이 문턱을 넘어 가짜 Ct가 나왔다
        # (가짜 판 스트레스 시험: 기기 Undetermined 웰에서 재계산 Ct 22.6·36.2 → DDI −13.8, NTC 오염 오판).
        # 판(표적) 고원 중앙값의 no_amp_frac배에 못 미치거나 기준선 잡음의 10배에 못 미치면 Ct를 내지 않는다
        pl = rise_t[target]
        pl_med = float(np.nanmax([pl.median() if len(pl) else np.nan, 0.5 * plate_med]))
        # 잡음 기준은 웰 자신의 기준선 SD가 아니라 판(표적)의 기준선 SD 중앙값: 기준선 1–3에서 이미 오르는 이른 웰은
        # 자기 SD가 커서 증폭 없음으로 잘못 찍혔다(실측 2차 판에서 4웰, 곡선 모양만 본 수)
        noise_med = float(gt.groupby("well")["baseline_sd"].first().median()) if "baseline_sd" in gt else 0.0
        for well, g in gt.groupby("well", sort=False):
            g = g.sort_values("cycle")
            c = g["cycle"].to_numpy(float)
            y = g["drn"].to_numpy(float)
            flags = []
            ymax = float(pl.get(well, np.nan))
            no_amp = no_amp_frac is not None and (not np.isfinite(ymax) or ymax < no_amp_frac * pl_med
                                                  or (np.isfinite(noise_med) and noise_med > 0 and ymax < 10 * noise_med))
            if not np.isfinite(thr):
                ct = float("nan")
                flags.append("no_threshold")
            elif no_amp:
                ct = float("nan")
                flags.append("no_amplification")
            else:
                ct = _crossing(c, y, thr)
            if np.isfinite(ct) and ct < hi + 5:
                flags.append("early_amp")
            position = posmap.get(well, "")
            if position in ("above_exponential", "below_noise", "not_reached"):
                flags.append("threshold_not_exponential")
            rows.append(dict(well=well, target=target, baseline_start=lo, baseline_end=hi, threshold=thr,
                             ct_recomputed=ct, undetermined_recomputed=not np.isfinite(ct),
                             plateau_drn=float(np.nanmax(y)) if len(y) else np.nan,
                             baseline_sd=float(g["baseline_sd"].iloc[0]),
                             baseline_slope=float(g["baseline_slope"].iloc[0]),
                             threshold_position=position, recompute_flags=";".join(flags)))
    out = pd.DataFrame(rows)
    if instrument is not None and len(out):
        inst = instrument[["well", "ct"]].rename(columns={"ct": "ct_instrument"}).drop_duplicates("well")
        out = out.merge(inst, on="well", how="left")
        out["ct_diff"] = out["ct_recomputed"] - out["ct_instrument"]
    return out


# --------------------------------------------------------------------------- 효율


def well_efficiency(amplification: pd.DataFrame, baseline_cycles=(3, 15), window: int = 4, lo_frac: float = 2e-3,
                    top_frac: float = 0.3, min_r2: float = 0.98) -> pd.DataFrame:
    """LinRegPCR 방식 웰별 효율: 기준선을 뺀 log10 ΔRn의 지수 구간에서 직선 창을 찾아 기울기로.

    - 기준선: 먼저 baseline_cycles로 빼고, 곡선이 오르기 시작하는 사이클을 찾아 그 4 사이클 전까지로
      다시 맞춘다(오름의 끝자락이 기준선에 섞이면 log 기울기가 휘기 때문. Ct 재계산의 공통 기준선과는 별개)

    - 창: 연속 `window` 사이클, 모든 점이 [하한, 상한] 안. 하한 = max(5 × 기준선 SD, 정체기 × 1e-4),
      상한 = 정체기 × top_frac (정체기에 가까울수록 기울기가 줄어 효율이 낮게 나오므로 아래쪽만 쓴다)
    - r² ≥ min_r2인 창 가운데 기울기가 가장 큰 창(지수 증폭은 아래쪽에서 가장 가파르다)
    - 증폭 배수 = 10^기울기, `efficiency` = 배수 − 1
    - 플래그: 배수 < 1.6 `eff_low`, > 2.1 `eff_high`, 창 없음 `no_window`
    반환 열: well, target, amp_factor, efficiency, window_start, window_end, n_points, r2, eff_flags
    """
    sub = subtract_baseline(amplification, baseline_cycles)
    rows = []
    for (well, target), g in sub.groupby(["well", "target"], sort=False):
        g = g.sort_values("cycle")
        c = g["cycle"].to_numpy(float)
        rn = g["rn"].to_numpy(float)
        y = g["drn"].to_numpy(float)
        sd = float(g["baseline_sd"].iloc[0]) if np.isfinite(g["baseline_sd"].iloc[0]) else 0.0
        plateau = float(np.nanmax(y)) if len(y) else np.nan
        y, sd = _refit_baseline_before_rise(c, rn, y, sd, plateau, baseline_cycles[0])
        best = None
        if np.isfinite(plateau) and plateau > 20 * sd and plateau > 0:
            lo_v = max(5 * sd, plateau * lo_frac)
            hi_v = plateau * top_frac
            ok = (y >= lo_v) & (y <= hi_v)
            for i in range(0, len(c) - window + 1):
                if not ok[i:i + window].all():
                    continue
                xs, ys = c[i:i + window], np.log10(y[i:i + window])
                slope, icpt = np.polyfit(xs, ys, 1)
                pred = slope * xs + icpt
                ss = np.sum((ys - ys.mean()) ** 2)
                r2 = 1 - np.sum((ys - pred) ** 2) / ss if ss > 0 else 0.0
                if r2 >= min_r2 and (best is None or slope > best[0]):
                    best = (slope, xs[0], xs[-1], r2)
        if best is None:
            rows.append(dict(well=well, target=target, amp_factor=np.nan, efficiency=np.nan,
                             window_start=np.nan, window_end=np.nan, n_points=0, r2=np.nan, eff_flags="no_window"))
            continue
        amp = float(10 ** best[0])
        fl = []
        if amp < AMP_FACTOR_RANGE[0]:
            fl.append("eff_low")
        if amp > AMP_FACTOR_RANGE[1]:
            fl.append("eff_high")
        rows.append(dict(well=well, target=target, amp_factor=amp, efficiency=amp - 1.0,
                         window_start=best[1], window_end=best[2], n_points=window, r2=float(best[3]),
                         eff_flags=";".join(fl)))
    return pd.DataFrame(rows, columns=["well", "target", "amp_factor", "efficiency", "window_start",
                                       "window_end", "n_points", "r2", "eff_flags"])


def _refit_baseline_before_rise(c, rn, y, sd, plateau, start):
    """오름 시작(ΔRn > max(10·SD, 정체기 × 0.2 %)) 4 사이클 전까지로 기준선을 다시 맞춘다."""
    if not np.isfinite(plateau) or plateau <= 0:
        return y, sd
    rise = np.where(y > max(10 * sd, 0.002 * plateau))[0]
    if len(rise) == 0:
        return y, sd
    end = c[rise[0]] - 4
    sel = (c >= start) & (c <= end) & np.isfinite(rn)
    if sel.sum() < 4:
        return y, sd
    coef = np.polyfit(c[sel], rn[sel], 1)
    resid = rn[sel] - np.polyval(coef, c[sel])
    return rn - np.polyval(coef, c), float(np.std(resid, ddof=2))


# --------------------------------------------------------------------------- ROX


def rox_diagnostics(multicomponent: pd.DataFrame, dye: str = "ROX", low_ratio: float = 0.5,
                    max_cv: float = 0.05, max_drift_frac: float = 0.10) -> pd.DataFrame:
    """웰별 ROX(기준 염료) 원시 형광 진단. BADROX의 원인을 가르는 근거 표.

    반환 열: well, rox_mean, rox_cv(사이클에 걸친 SD/평균), rox_drift_per_cycle(직선 기울기/평균, 사이클당 비율),
    rox_drift_total(기울기 × 사이클 수 / 평균), rox_ratio_to_median(판 중앙값 대비), rox_flags
    플래그: `rox_low`(비 < low_ratio), `rox_unstable`(CV > max_cv), `rox_drift`(|총 표류| > max_drift_frac),
    `rox_missing`(ROX 열 없음)
    기준값(0.5, 5 %, 10 %)은 점검용 기본값이다. 확정 기준은 사람이 정한다(리포트 "사람이 결정할 것").
    """
    d = multicomponent[multicomponent["dye"].astype(str).str.upper() == dye.upper()]
    cols = ["well", "rox_mean", "rox_cv", "rox_drift_per_cycle", "rox_drift_total", "rox_ratio_to_median", "rox_flags"]
    if d.empty:
        wells = multicomponent["well"].drop_duplicates()
        return pd.DataFrame({"well": wells, "rox_mean": np.nan, "rox_cv": np.nan, "rox_drift_per_cycle": np.nan,
                             "rox_drift_total": np.nan, "rox_ratio_to_median": np.nan, "rox_flags": "rox_missing"})[cols]
    rows = []
    for well, g in d.groupby("well", sort=False):
        g = g.sort_values("cycle")
        f = g["fluorescence"].to_numpy(float)
        c = g["cycle"].to_numpy(float)
        m = float(np.nanmean(f))
        cv = float(np.nanstd(f, ddof=1) / m) if len(f) > 1 and m != 0 else np.nan
        slope = float(np.polyfit(c, f, 1)[0]) if len(f) > 1 else np.nan
        rows.append(dict(well=well, rox_mean=m, rox_cv=cv, rox_drift_per_cycle=slope / m if m else np.nan,
                         rox_drift_total=slope * (c.max() - c.min()) / m if m else np.nan))
    out = pd.DataFrame(rows)
    med = float(np.nanmedian(out["rox_mean"]))
    out["rox_ratio_to_median"] = out["rox_mean"] / med if med else np.nan
    flags = []
    for r in out.itertuples(index=False):
        fl = []
        if r.rox_ratio_to_median < low_ratio:
            fl.append("rox_low")
        if r.rox_cv > max_cv:
            fl.append("rox_unstable")
        if abs(r.rox_drift_total) > max_drift_frac:
            fl.append("rox_drift")
        flags.append(";".join(fl))
    out["rox_flags"] = flags
    return out[cols]


def rox_plate_evidence(rox: pd.DataFrame, instrument_flags: pd.Series | None = None,
                       passive_reference: str | None = None, expected_rox_min: float | None = None) -> pd.DataFrame:
    """BADROX가 "ROX 신호 자체의 문제"인지 "설정 문제"인지 가를 근거를 표로 모은다.

    원인을 단정하지 않는다. 각 줄은 `항목, 값, 어느 쪽 근거인지(가리키는 쪽)`.
    - 웰 대부분이 rox_low/unstable이면 → ROX 신호 쪽
    - BADROX 웰이 많은데 ROX가 판 전체에서 고르고 안정적이면 → 설정(Passive Reference·투입량 불일치) 쪽
    - `expected_rox_min`: 기기·키트 기준 ROX 최소 수준(모르면 None → 절대 수준은 판정하지 않음)
    """
    n = len(rox)
    fl = rox["rox_flags"].fillna("")
    n_low = int(fl.str.contains("rox_low").sum())
    n_unst = int(fl.str.contains("rox_unstable").sum())
    n_drift = int(fl.str.contains("rox_drift").sum())
    plate_cv = float(np.nanstd(rox["rox_mean"], ddof=1) / np.nanmean(rox["rox_mean"])) if n > 1 else np.nan
    med_cv = float(np.nanmedian(rox["rox_cv"])) if n else np.nan
    rows = [
        ("웰 수", n, ""),
        ("ROX 판 중앙값(원시 형광)", float(np.nanmedian(rox["rox_mean"])) if n else np.nan, ""),
        ("웰 간 ROX 평균의 변동계수", plate_cv, "높으면 ROX 신호 쪽(웰마다 투입량·기포 차이)"),
        ("웰 안 사이클 CV 중앙값", med_cv, "높으면 ROX 신호 쪽(흔들림)"),
        ("rox_low 웰 수", n_low, "많으면 ROX 신호 쪽"),
        ("rox_unstable 웰 수", n_unst, "많으면 ROX 신호 쪽"),
        ("rox_drift 웰 수", n_drift, "많으면 ROX 신호 쪽(증발·광표백)"),
    ]
    if instrument_flags is not None:
        n_bad = int(instrument_flags.fillna("").str.contains("BADROX").sum())
        rows.append(("기기 BADROX 웰 수", n_bad, ""))
        if n_bad > 0 and n_low + n_unst == 0:
            rows.append(("BADROX인데 원시 ROX가 고르고 안정", True, "설정 쪽(Passive Reference 설정과 실제 첨가 불일치) 근거"))
    rows.append(("Passive Reference 설정", passive_reference if passive_reference is not None else "모름", "실험 노트의 ROX 투입량과 대조"))
    if expected_rox_min is not None and n:
        below = float(np.nanmedian(rox["rox_mean"])) < expected_rox_min
        rows.append((f"판 중앙값 < 기대 최소({expected_rox_min:g})", below, "참이면 ROX 투입량 부족 또는 미첨가 쪽"))
    else:
        rows.append(("ROX 절대 수준 기준", "없음", "기기·키트의 기대 ROX 수준을 사람이 정해야 판정 가능"))
    return pd.DataFrame(rows, columns=["항목", "값", "가리키는 쪽"])


# --------------------------------------------------------------------------- 멜트


def melt_summary(melt: pd.DataFrame, min_rel_height: float = 0.15, min_abs_frac: float = 0.10,
                 min_separation_C: float = 1.5, smooth: int = 3) -> pd.DataFrame:
    """웰별 멜트 미분 곡선(−dF/dT)의 피크 수와 Tm. 피크 2개 이상이면 비특이 증폭(프라이머 이합체 등) 의심.

    - 피크: 웰 최대의 min_rel_height 이상 돌출(prominence), 서로 min_separation_C °C 이상 떨어짐
    - 신호가 판 전체 웰 최대값 중앙값의 min_abs_frac보다 작은 웰(평평한 NTC 등)은 피크 0개
    - smooth: 이동 평균 창(점 수). 1이면 끔
    반환 열: well, n_peaks, tm1, tm2, peak_height1, peak_height2, melt_flags(`multi_peak`, `no_peak`)
    tm1은 가장 높은 피크의 온도(°C)
    """
    from scipy.signal import find_peaks

    per = []
    for well, g in melt.groupby("well", sort=False):
        g = g.sort_values("temperature_C")
        d = g["derivative"].to_numpy(float)
        if smooth > 1 and len(d) >= smooth:
            d = pd.Series(d).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
        # 배경(완만한 기울기)을 빼고 피크를 본다
        d = d - np.nanmin(d)
        per.append((well, g["temperature_C"].to_numpy(float), d))
    maxima = [np.nanmax(d) if len(d) else 0.0 for _, _, d in per]
    ref = float(np.nanmedian(maxima)) if maxima else 0.0
    rows = []
    for (well, t, d), mx in zip(per, maxima):
        tm = [np.nan, np.nan]
        hts = [np.nan, np.nan]
        if len(d) < 3 or mx <= 0 or mx < min_abs_frac * ref:
            n = 0
        else:
            step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
            idx, props = find_peaks(d, prominence=min_rel_height * mx,
                                    distance=max(1, int(round(min_separation_C / step))))
            order = np.argsort(d[idx])[::-1]
            idx = idx[order]
            n = len(idx)
            for k in range(min(2, n)):
                tm[k] = float(t[idx[k]])
                hts[k] = float(d[idx[k]])
        fl = "multi_peak" if n >= 2 else ("no_peak" if n == 0 else "")
        rows.append(dict(well=well, n_peaks=n, tm1=tm[0], tm2=tm[1], peak_height1=hts[0],
                         peak_height2=hts[1], melt_flags=fl))
    return pd.DataFrame(rows, columns=["well", "n_peaks", "tm1", "tm2", "peak_height1", "peak_height2", "melt_flags"])


def threshold_scan(drn_table: pd.DataFrame, thresholds) -> pd.DataFrame:
    """문턱 격자에서 곡선이 얼마나 나란한지 본다.

    문턱마다: 검출 웰 수, Ct 중앙값, 다음 문턱과의 웰별 Ct 차이의 SD(작을수록 그 구간에서 곡선이 나란함),
    다음 문턱과의 Ct 순위 상관. 결과(시료·조건)가 아니라 곡선 모양만 쓴다. 입력은 subtract_baseline 결과.
    반환 열: threshold, n_detected, ct_median, sd_diff_next, spearman_next
    """
    ths = sorted(float(t) for t in thresholds)
    cts = {}
    for t in ths:
        vals = {}
        for well, g in drn_table.groupby("well", sort=False):
            g = g.sort_values("cycle")
            vals[well] = _crossing(g["cycle"].to_numpy(float), g["drn"].to_numpy(float), t)
        cts[t] = pd.Series(vals)
    rows = []
    for i, t in enumerate(ths):
        c = cts[t]
        sd = rho = np.nan
        if i + 1 < len(ths):
            pair = pd.concat([c, cts[ths[i + 1]]], axis=1).dropna()
            if len(pair) > 2:
                sd = float((pair.iloc[:, 1] - pair.iloc[:, 0]).std())
                rho = float(pair.corr(method="spearman").iloc[0, 1])
        rows.append(dict(threshold=t, n_detected=int(c.notna().sum()), ct_median=float(c.median()),
                         sd_diff_next=sd, spearman_next=rho))
    return pd.DataFrame(rows)
