"""가짜 StepOne 내보내기 픽스처를 만든다. **모든 파일은 가짜(`fake_` 이름)이며 실측이 아니다.**

실제 StepOne Software v2.3 Results 내보내기의 모양(메타데이터 줄, 열 이름, 경고 열)을
문서화하는 용도다. 다시 만들 때만 실행한다(.xls 쓰기에 `xlwt` 필요, 읽기는 xlrd):

    python tests/qpcr/fixtures/make_fake_fixtures.py

형식 세 가지:
1. fake_results_v23.xls — 메타데이터 줄 있음, Ct 머리줄이 키릴 문자 `Cт`, 경고가 개별 Y/N 열
2. fake_results_nometa.txt — 메타데이터 없음, 탭 구분, `CT` 머리줄, 경고가 `Flag` 한 열, Well이 숫자
3. fake_results_sections.txt — 메타데이터 + `[Results]`·`[Amplification Data]` 구역이 한 파일에
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

META = [
    "* Block Type = 96-Well Block (0.1mL)",
    "* Chemistry = SYBR_GREEN",
    "* Experiment File Name = C:\\Applied Biosystems\\StepOne Software v2.3\\experiments\\FAKE_plate.eds",
    "* Experiment Run End Time = 2026-11-06 09:00:00 AM KST",
    "* Instrument Type = steponeplus",
    "* Passive Reference = ROX",
]

# well, sample, target, task, ct, quantity, tm1, tm2, flags
WELLS = [
    ("A1", "C-080-30-1", "COI-104", "UNKNOWN", "21.40", "", "79.8", "", ""),
    ("A2", "C-080-30-1", "COI-104", "UNKNOWN", "21.55", "", "79.9", "", ""),
    ("A3", "C-080-30-1", "COI-586", "UNKNOWN", "24.10", "", "84.6", "", ""),
    ("A4", "C-080-30-1", "COI-586", "UNKNOWN", "Undetermined", "", "", "", "NOAMP"),
    ("B1", "STD-S104-1e4", "COI-104", "STANDARD", "18.02", "10000", "79.8", "", ""),
    ("B2", "STD-S104-1e3", "COI-104", "STANDARD", "21.35", "1000", "79.8", "", ""),
    ("B3", "NTC-1", "COI-104", "NTC", "Undetermined", "", "", "", ""),
    ("B4", "NTC-2", "COI-586", "NTC", "36.80", "", "74.2", "", ""),
    ("C1", "RAW-B1-1", "COI-104", "UNKNOWN", "20.90", "", "79.7", "74.1", "BADROX;EXPFAIL"),
    ("C2", "RAW-B1-1", "mystery-target", "UNKNOWN", "25.00", "", "84.5", "", "BADROX"),
]

FLAG_NAMES = ["BADROX", "EXPFAIL", "NOAMP", "HIGHSD"]


def write_v23_xls(path: Path) -> None:
    import xlwt

    wb = xlwt.Workbook(encoding="utf-8")
    ws = wb.add_sheet("Results")
    r = 0
    for line in META:
        ws.write(r, 0, line)
        r += 1
    r += 1
    header = ["Well", "Sample Name", "Target Name", "Task", "Reporter", "Quencher", "C\u0442",
              "C\u0442 Mean", "C\u0442 SD", "Quantity", "Quantity Mean", "Automatic Ct Threshold",
              "Ct Threshold", "Automatic Baseline", "Baseline Start", "Baseline End", "Comments",
              *FLAG_NAMES, "Tm1", "Tm2", "Tm3"]
    for c, h in enumerate(header):
        ws.write(r, c, h)
    r += 1
    for well, sample, target, task, ct, qty, tm1, tm2, flags in WELLS:
        fl = flags.split(";") if flags else []
        vals = [well, sample, target, task, "SYBR", "None", _num(ct), "", "", _num(qty), "",
                "true", 0.2, "true", 3, 15, "",
                *["Y" if f in fl else "N" for f in FLAG_NAMES], _num(tm1), _num(tm2), ""]
        for c, v in enumerate(vals):
            ws.write(r, c, v)
        r += 1
    wb.save(str(path))


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return s


def write_nometa_txt(path: Path) -> None:
    header = ["Well", "Well Position", "Sample Name", "Target Name", "Task", "Reporter", "CT",
              "Quantity", "Tm1", "Tm2", "Flag"]
    lines = ["\t".join(header)]
    for i, (well, sample, target, task, ct, qty, tm1, tm2, flags) in enumerate(WELLS):
        num = (ord(well[0]) - ord("A")) * 12 + int(well[1:])
        lines.append("\t".join([str(num), well, sample, target, task, "SYBR", ct, qty, tm1, tm2,
                                flags.replace(";", ", ")]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sections_txt(path: Path) -> None:
    lines = list(META) + ["", "[Amplification Data]",
                          "Well\tCycle\tTarget Name\tRn\tDelta Rn"]
    for cyc in range(1, 4):
        lines.append(f"A1\t{cyc}\tCOI-104\t{1.0 + 0.001 * cyc:.4f}\t{0.001 * cyc:.4f}")
    lines += ["", "[Results]",
              "Well\tSample Name\tTarget Name\tTask\tReporter\tC\u0442\tQuantity\tComments\tBADROX\tEXPFAIL"]
    for well, sample, target, task, ct, qty, tm1, tm2, flags in WELLS[:4]:
        fl = flags.split(";") if flags else []
        lines.append("\t".join([well, sample, target, task, "SYBR", ct, qty, "",
                                "Y" if "BADROX" in fl else "N", "Y" if "EXPFAIL" in fl else "N"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_v23_xls(HERE / "fake_results_v23.xls")
    write_nometa_txt(HERE / "fake_results_nometa.txt")
    write_sections_txt(HERE / "fake_results_sections.txt")
    print("만듦:", sorted(p.name for p in HERE.glob("fake_*")))
