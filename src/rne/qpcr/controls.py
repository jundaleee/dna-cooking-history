"""0단계 대조 칸과 특이성 판정: 재료를 하나씩 뺀 대조 칸(주형 없음 NTC, 프라이머 없음, 믹스만, 물만)을
**기기 Ct가 아니라 원신호(Multicomponent SYBR)와 멜트 모양**으로 판정한다.

원신호 특징(웰마다, `well_signal_features`):
- rise_snr: (SYBR 마지막 3 cycle 평균 − 1–10 cycle 직선 기준선을 늘린 값) / (그 늘린 값의 표준오차).
  기준선 기울기는 상승으로 세지 않는다. 오르지 않은 웰은 표준정규 근처
- climb_snr: 앞선 최저점에서 가장 크게 오른 양 / 1차 차분 강건 잡음. 초반 형광 감소는 오름을 만들지 않는다
- fold: 끝 3 cycle / 처음 3 cycle SYBR 비(아주 이른 웰을 놓치지 않으려는 보조)
- onset_cycle: SYBR가 처음 3 cycle 평균 + 상승의 절반을 처음 넘는 cycle(오른 웰만)
- melt_step_frac: 멜트 구간 어느 5 °C 창에서든 직선(전 구간 맞춤)보다 더 떨어진 양의 최대 / 증폭 동안 오른 SYBR.
  표적 이중가닥이 녹으면 크다
- peak_T, peak_prom: SYBR −dF/dT 최대 온도와 (최대 / 중앙값)

판정(웰 종류별, `judge_controls`):
- 물만·믹스만·프라이머 없음: 오르지 않아야 통과. 프라이머 없음 웰이 오르면 "프라이머와 무관한 신호" → 실패
- NTC: 오르지 않으면 통과. 오르면 멜트 봉우리가 표적 Tm 창 안이고 계단이 있으면 "주형 오염" → 실패,
  창 밖이거나 계단이 없으면 "비특이(이합체 등) 신호" → 주의. 단, 오른 NTC의 onset이 시료 onset 중앙값 + ntc_min_gap
  보다 이르면 실패(시료 신호와 구별 안 됨)
- 시료 웰 특이성(`specificity`): 오른 시료 웰 중 계단 ≥ 상승 × step_min_frac이고 봉우리가 표적 Tm 창 안인 웰의 비율이
  site_min_frac 이상이면 "표적 확인(멜트)", 아니면 "표적 미확인"(측정이 정상이 아님)

모든 숫자 기준(`DEFAULT_SETTINGS`)은 측정 쪽 **제안**이고 사전 고정에서 사람이 정한다. 표적 Tm 창은 실측 근거가 없어
기본값이 None(창 검사 생략, 계단만 봄)이다. 전기영동 띠 크기 허용 범위는 `gel_band_check`.

    python -m rne.qpcr.controls --export <판 .xls> --layout <배치표.csv> --out <폴더>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rne.qpcr import raw

#: 측정 쪽 제안 기준(사람이 사전 고정에서 정한다)
DEFAULT_SETTINGS = dict(
    rise_snr_min=10.0,          # 이 값 이상이면 "오름"
    climb_snr_min=10.0,         # 그리고 앞선 최저점에서 오른 양 / 차분 잡음이 이 값 이상(초반 감소를 오름으로 읽지 않음)
    climb_rel_min=0.02,         # 그리고 오른 양 / 처음 신호가 이 값 이상(J6: 평평한 웰 A3 0.009의 climb_snr 16 오판 방지,
                                # 약한 늦은 증폭 5 %는 잡는다)
    fold_min=1.5,               # 또는 끝/처음 SYBR 비가 이 값 이상이면 "오름"(아주 이른 웰)
    step_min_frac=0.25,         # 멜트 계단 / 상승이 이 값 이상이면 "녹는 산물 있음"
    window_C=5.0,               # 계단을 재는 온도 창
    tm_window_C=None,           # {"S104": [lo, hi], "L586": [lo, hi]} 또는 None(창 검사 생략)
    ntc_min_gap=5.0,            # 오른 NTC onset이 시료 onset 중앙값 + 이 값보다 이르면 실패
    site_min_frac=0.5,          # 시료 웰 중 녹는 산물이 있는 웰이 이 비율 이상이어야 "표적 확인"
    max_ntc_warn=2,             # 비특이 신호 NTC가 이 개수를 넘으면 판 상태를 "주의"에서 "실패"로
    gel_tolerance_frac=0.2,     # 띠 크기 허용: 기대 bp × (1 ± 이 값)
)
EXPECTED_BP = {"S104": 104, "L586": 586}
CLIMB_WINDOW = 8  # climb_snr를 재는 cycle 창(증폭 곡선의 10–90 % 상승은 약 7 cycle)
NEGATIVE_KINDS = ("water", "mixonly", "noprimer")


def _settings(s: dict | None) -> dict:
    out = dict(DEFAULT_SETTINGS)
    out.update(s or {})
    return out


def curve_features(s: np.ndarray) -> dict:
    """증폭 구간 SYBR 원신호 하나(cycle 1부터, 12개 이상) → sybr_start, sybr_rise, rise_snr, climb_snr, fold, onset_cycle."""
    s = np.asarray(s, float)
    cyc = np.arange(1, len(s) + 1, dtype=float)
    bw = slice(0, 10)  # 기준선: 1–10 cycle 직선(기준선 기울기를 상승으로 세지 않는다)
    coef = np.polyfit(cyc[bw], s[bw], 1)
    line = np.polyval(coef, cyc)
    res = s[bw] - line[bw]
    noise = float(np.std(res, ddof=2)) if np.std(res) > 0 else 1e-12
    base, end = float(s[:3].mean()), float(s[-3:].mean())
    rise = float((s[-3:] - line[-3:]).mean())
    # 기준선을 30 cycle 넘게 밖으로 늘린 값의 표준오차(기울기 오차가 커진다)로 나눈다. 잡음 sd로만 나누면
    # 평평한 웰이 우연히 10을 넘는다
    xb = cyc[bw]
    se_pred = noise * np.sqrt(1 / 3 + 1 / len(xb) + (cyc[-3:].mean() - xb.mean()) ** 2 / ((xb - xb.mean()) ** 2).sum())
    rise_snr = rise / max(se_pred, 1e-12 * max(abs(base), 1.0))
    fold = end / base if base > 0 else np.nan
    # climb_snr: 앞선 최저점에서 가장 크게 오른 양 / 1차 차분의 강건 잡음(MAD). 초반 형광 감소(광표백 등)는 오름을 만들지
    # 않으므로 기준선 직선 외삽이 아래로 휘어 rise_snr가 커져도 이 값은 작다
    # 기준선 기울기는 차분 중앙값으로 빼고(증폭은 몇 cycle에만 몰린다), climb_window cycle 안에서 오른 양만 센다
    dif = np.diff(s)
    med = float(np.median(dif))
    noise_d = 1.4826 * float(np.median(np.abs(dif - med))) / np.sqrt(2.0)
    sd_ = s - max(med, 0.0) * cyc  # 내려가는 기울기(초반 감소의 꼬리)는 빼지 않는다: 빼면 평평한 뒤쪽이 올라 보인다
    w = CLIMB_WINDOW
    climb = float(max(sd_[j] - sd_[max(0, j - w):j + 1].min() for j in range(len(sd_))))
    climb_snr = climb / max(noise_d, 1e-12 * max(abs(base), 1.0))
    # climb_rel: 오른 양 / 처음 3 cycle 평균. 신호가 양자화된 평평한 웰(2차 판 A3, fold 0.99)은 차분 MAD가 0이라
    # climb_snr가 한 눈금으로 16이 됐다 → 오름 판정에 상대 크기 바닥(climb_rel_min)을 함께 둔다
    climb_rel = climb / abs(base) if abs(base) > 0 else np.inf
    above = s - line
    half = 0.5 * rise
    onset = float(np.argmax(above >= half) + 1) if rise > 0 and (above >= half).any() else np.nan
    return dict(sybr_start=base, sybr_rise=rise, rise_snr=rise_snr, climb_snr=climb_snr, climb_rel=climb_rel, fold=fold,
                onset_cycle=onset)


def well_signal_features(path, window_C: float = 5.0) -> pd.DataFrame:
    """내보내기 파일 하나 → 웰마다 원신호·멜트 특징 표(열은 모듈 설명). 위치·조건은 쓰지 않는다."""
    mc = raw.read_multicomponent(path, max_cycle=None)
    try:
        amp_max = float(raw.read_amplification(path)["cycle"].max())
    except Exception:  # noqa: BLE001 - Amplification Data가 없으면 40 cycle로 본다
        amp_max = 40.0
    try:
        melt = raw.read_melt(path)
    except Exception:  # noqa: BLE001
        melt = None
    rows = []
    for w, g in mc.groupby("well"):
        dyes = {d: s.sort_values("cycle") for d, s in g.groupby("dye")}
        key = next((k for k in dyes if "SYBR" in k), None)
        if key is None:
            continue
        s_all = dyes[key]
        s = s_all[s_all["cycle"] <= amp_max]["fluorescence"].to_numpy(float)
        if len(s) < 12:
            continue
        rec = dict(well=w, **curve_features(s), melt_step_frac=np.nan, peak_T=np.nan, peak_prom=np.nan)
        rise = rec["sybr_rise"]
        if melt is not None:
            m = melt[melt["well"] == w].sort_values("temperature_C")
            k = len(m)
            tail = s_all[s_all["cycle"] > amp_max]["fluorescence"].to_numpy(float)
            if k >= 10 and len(tail) >= k:
                T = m["temperature_C"].to_numpy(float)
                f = tail[-k:]
                a, b = np.polyfit(T, f, 1)
                best = 0.0
                for i in range(k):
                    j = np.searchsorted(T, T[i] + window_C)
                    if j >= k:
                        break
                    excess = (f[i] - f[j]) - (-a) * (T[j] - T[i])
                    best = max(best, excess)
                d = -np.gradient(f, T)
                rec.update(melt_step_frac=best / rise if rise > 0 else np.nan, peak_T=float(T[np.argmax(d)]),
                           peak_prom=float(d.max() / max(np.median(np.abs(d)), 1e-12)))
        rows.append(rec)
    return pd.DataFrame(rows)


def _in_window(peak_T, target, settings) -> bool | None:
    win = (settings.get("tm_window_C") or {}).get(str(target).strip().upper()) if settings.get("tm_window_C") else None
    if win is None or not np.isfinite(peak_T):
        return None
    return bool(win[0] <= peak_T <= win[1])


def _melting(r, target, settings) -> bool:
    """녹는 산물(계단 있음, 그리고 창이 정해졌으면 봉우리가 창 안)."""
    step = r["melt_step_frac"]
    ok = bool(np.isfinite(step) and step >= settings["step_min_frac"])
    inw = _in_window(r["peak_T"], target, settings)
    return ok and (inw is None or inw)


def rose_mask(t: pd.DataFrame, st: dict) -> pd.Series:
    """'오름': 기준선 위 상승 rise_snr ≥ rise_snr_min, 또는 처음 대비 끝 형광 비 fold ≥ fold_min.

    둘째 조건은 아주 이른 웰(기준선 창 안에서 이미 오른 웰)을 '오르지 않음'으로 놓치지 않으려는 것이다(대조 웰의 거짓 통과 방지).
    """
    snr = t["rise_snr"].astype(float) >= st["rise_snr_min"]
    if "climb_snr" in t and st.get("climb_snr_min") is not None:
        climb = t["climb_snr"].astype(float) >= st["climb_snr_min"]
        if "climb_rel" in t and st.get("climb_rel_min") is not None:
            climb = climb & (t["climb_rel"].astype(float) >= st["climb_rel_min"])
        # "and": 기준선 위 상승과 실제로 오른 양이 둘 다
        # "climb": 기울기를 뺀 창 안 오름만
        snr = climb if st.get("rise_rule", "climb") == "climb" else (snr & climb)
    fold = t["fold"].astype(float) >= st["fold_min"] if "fold" in t else False
    return (snr | fold).fillna(False)


def judge_controls(features: pd.DataFrame, layout: pd.DataFrame, settings: dict | None = None) -> dict:
    """대조 웰 판정과 시료 웰 특이성. layout은 well, target, kind(sample/ntc/noprimer/mixonly/water) 열.

    반환: status(통과/주의/실패), reasons, wells(웰별 판정 표, records), specificity(dict), settings.
    """
    st = _settings(settings)
    lay = layout[layout["kind"].astype(str) != "empty"].copy()  # 빈칸 웰은 판정하지 않는다(plate5 예산 배치)
    lay["well"] = lay["well"].astype(str).str.strip().str.upper()
    f = features.copy()
    f["well"] = f["well"].astype(str).str.strip().str.upper()
    t = lay.merge(f, on="well", how="left")
    t["rose"] = rose_mask(t, st)
    samp = t[t["kind"] == "sample"]
    onset_med = float(samp.loc[samp["rose"], "onset_cycle"].median()) if samp["rose"].any() else np.nan
    verdicts, reasons = [], []
    n_ntc_warn = 0
    for _, r in t.iterrows():
        k = r["kind"]
        if k == "sample":
            verdicts.append("")
            continue
        if pd.isna(r["rise_snr"]):
            verdicts.append("실패: 신호 없음(파일에 웰 없음)")
            reasons.append(f"{r['well']} {k}: 원신호가 없다")
            continue
        if not r["rose"]:
            verdicts.append("통과")
            continue
        if k in NEGATIVE_KINDS:
            what = {"water": "물만", "mixonly": "믹스만", "noprimer": "프라이머 없음"}[k]
            verdicts.append("실패: 오름")
            reasons.append(f"{r['well']} {what} 웰이 올랐다(rise_snr {r['rise_snr']:.0f})"
                           + (" — 프라이머와 무관한 신호" if k == "noprimer" else ""))
            continue
        # NTC가 오름
        early = np.isfinite(onset_med) and np.isfinite(r["onset_cycle"]) and r["onset_cycle"] < onset_med + st["ntc_min_gap"]
        if early:
            verdicts.append("실패: 시료와 구별 안 되는 NTC")
            reasons.append(f"{r['well']} NTC onset {r['onset_cycle']:.0f} < 시료 중앙값 {onset_med:.0f} + {st['ntc_min_gap']:g}")
        elif _melting(r, r["target"], st) and _in_window(r["peak_T"], r["target"], st) is True:
            verdicts.append("실패: 주형 오염 의심")
            reasons.append(f"{r['well']} NTC가 표적 Tm 창 안 봉우리({r['peak_T']:.1f} °C)로 올랐다 — 오염 의심(미확인)")
        else:
            verdicts.append("주의: 비특이 신호")
            n_ntc_warn += 1
    t["verdict"] = verdicts
    fails = [v for v in verdicts if v.startswith("실패")]
    if n_ntc_warn > st["max_ntc_warn"]:
        reasons.append(f"비특이 신호 NTC {n_ntc_warn}개 > {st['max_ntc_warn']}")
    status = "실패" if fails or n_ntc_warn > st["max_ntc_warn"] else ("주의" if n_ntc_warn else "통과")
    if n_ntc_warn and status == "주의":
        reasons.append(f"비특이 신호 NTC {n_ntc_warn}개(늦게 오름, 표적 봉우리 아님)")
    spec = specificity(t[t["kind"] == "sample"], st)
    cols = ["well", "target", "kind", "rise_snr", "climb_snr", "fold", "onset_cycle", "melt_step_frac", "peak_T", "verdict"]
    cols = [c for c in cols if c in t]
    return dict(status=status, reasons=reasons, n_controls=int((t["kind"] != "sample").sum()),
                wells=t[t["kind"] != "sample"][cols].replace({np.nan: None}).to_dict("records"),
                specificity=spec, settings=st)


def specificity(samples: pd.DataFrame, settings: dict | None = None) -> dict:
    """시료 웰의 멜트 특이성: 오른 웰 중 녹는 산물이 있는 웰의 비율(표적별). 기준 미달이면 confirmed=False."""
    st = _settings(settings)
    out = dict(by_target={}, confirmed=True, reasons=[], melt_available=True)
    for tg, g in samples.groupby(samples["target"].astype(str).str.strip().str.upper()):
        rose = g[rose_mask(g, st)]
        if len(rose) and rose["melt_step_frac"].isna().all():
            # 멜트 자료가 없으면 '표적 미확인'(측정 이상)이 아니라 '판정할 자료 없음'이다
            out["melt_available"] = False
            out["reasons"].append(f"{tg}: 멜트 자료 없음 — 내보내기 형식 확인")
        n_melt = int(sum(_melting(r, tg, st) for _, r in rose.iterrows()))
        frac = n_melt / len(rose) if len(rose) else 0.0
        out["by_target"][tg] = dict(n_wells=int(len(g)), n_rose=int(len(rose)), n_melting=n_melt, frac=frac,
                                    median_step=float(rose["melt_step_frac"].median()) if len(rose) else None,
                                    median_peak_T=float(rose["peak_T"].median()) if len(rose) else None)
        if frac < st["site_min_frac"]:
            out["confirmed"] = False
            out["reasons"].append(f"{tg}: 녹는 산물이 있는 웰 {n_melt}/{len(rose)} < {st['site_min_frac']:g} → 표적 미확인")
    return out


def gel_band_check(bands: pd.DataFrame, settings: dict | None = None) -> pd.DataFrame:
    """전기영동 띠 크기 판정. bands 열: lane, target, band_bp(띠 하나면 한 줄, 여럿이면 여러 줄).

    target별 기대 크기 × (1 ± gel_tolerance_frac) 안의 띠가 정확히 하나이고 그 밖의 띠가 없으면 '맞음'.
    반환: lane, target, n_bands, n_in_range, verdict. 레인 정보가 없는 지금은 함수만 둔다.
    """
    st = _settings(settings)
    rows = []
    for (lane, tg), g in bands.groupby(["lane", "target"]):
        exp = EXPECTED_BP.get(str(tg).strip().upper())
        sizes = g["band_bp"].dropna().astype(float)
        if exp is None:
            rows.append(dict(lane=lane, target=tg, n_bands=len(sizes), n_in_range=0, verdict="기대 크기 없음"))
            continue
        lo, hi = exp * (1 - st["gel_tolerance_frac"]), exp * (1 + st["gel_tolerance_frac"])
        n_in = int(((sizes >= lo) & (sizes <= hi)).sum())
        v = "맞음" if n_in == 1 and len(sizes) == 1 else ("띠 없음" if len(sizes) == 0 else
                                                           ("크기 다름" if n_in == 0 else "띠 여럿"))
        rows.append(dict(lane=lane, target=tg, n_bands=len(sizes), n_in_range=n_in, verdict=v))
    return pd.DataFrame(rows)


def plot_control_curves(export_path, layout_path, path, settings: dict | None = None) -> str:
    """대조 웰마다 SYBR 원신호(증폭 구간)를 판정과 함께 그린다. 시료 웰은 중앙값과 10–90 % 띠만(위치·조건 없음).

    동결 전 수령(`resultday receive`)에서 사람이 숫자 기준을 정하기 전에 원신호를 직접 보게 한다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 한글 범례가 네모로 깨지지 않게 — 있는 한글 글꼴을 쓴다
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ("Malgun Gothic", "NanumGothic", "AppleGothic") if f in have] + ["DejaVu Sans"]
    st = _settings(settings)
    lay = pd.read_csv(layout_path, encoding="utf-8-sig")
    mc = raw.read_multicomponent(export_path, max_cycle=None)
    try:
        amp_max = float(raw.read_amplification(export_path)["cycle"].max())
    except Exception:  # noqa: BLE001
        amp_max = 40.0
    mc = mc[mc["dye"].astype(str).str.contains("SYBR") & (mc["cycle"] <= amp_max)]
    feats = well_signal_features(export_path, st["window_C"])
    rose = dict(zip(feats["well"], rose_mask(feats, st)))
    kinds = dict(zip(lay["well"], lay["kind"].astype(str)))
    ctrl = [w for w in lay["well"] if kinds.get(w) not in ("sample", "empty")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=130)
    smp = mc[mc["well"].map(kinds).eq("sample")].pivot_table(index="cycle", columns="well", values="fluorescence")
    for ax in axes:
        if not smp.empty:
            ax.fill_between(smp.index, smp.quantile(0.1, axis=1), smp.quantile(0.9, axis=1), color="0.85", label="시료 10–90 %")
            ax.plot(smp.index, smp.median(axis=1), color="0.5", lw=1, label="시료 중앙값")
    colors = {"ntc": "tab:red", "noprimer": "tab:purple", "mixonly": "tab:blue", "water": "tab:green"}
    for w in ctrl:
        g = mc[mc["well"] == w].sort_values("cycle")
        if g.empty:
            continue
        k = kinds[w]
        ax = axes[0] if k == "ntc" else axes[1]
        ax.plot(g["cycle"], g["fluorescence"], color=colors.get(k, "k"), lw=1, ls="-" if rose.get(w) else "--",
                label=f"{w} {k}{' 오름' if rose.get(w) else ''}")
    for ax, t in zip(axes, ("NTC", "물만·믹스만·프라이머 없음")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("cycle")
        ax.set_ylabel("SYBR 원신호")
        ax.legend(fontsize=5, frameon=False, ncol=2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def check_plate(export_path, layout_path, settings: dict | None = None) -> dict:
    """한 판: 파일 → 특징 → 대조 웰 판정. 반환은 judge_controls와 같고 features(표)를 더한다."""
    lay = pd.read_csv(layout_path, encoding="utf-8-sig")
    st = _settings(settings)
    feats = well_signal_features(export_path, st["window_C"])
    res = judge_controls(feats, lay, st)
    res["features"] = feats
    return res


def main(argv=None) -> int:
    """명령줄: 대조 웰 판정을 json과 csv로 쓴다. 통과 0, 주의 0, 실패 1."""
    ap = argparse.ArgumentParser(prog="python -m rne.qpcr.controls")
    ap.add_argument("--export", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--settings", default=None, help="기준 JSON(없으면 DEFAULT_SETTINGS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    st = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else None
    res = check_plate(a.export, a.layout, st)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.pop("features").to_csv(out / "well_features.csv", index=False, encoding="utf-8-sig")
    (out / "controls.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"대조 웰: {res['status']} — " + "; ".join(res["reasons"]))
    print(f"특이성: {'표적 확인(멜트)' if res['specificity']['confirmed'] else '표적 미확인'}")
    return 1 if res["status"] == "실패" else 0


if __name__ == "__main__":
    raise SystemExit(main())
