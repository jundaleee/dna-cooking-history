"""PL3 코드 동결: 사전 고정 때 판정에 쓰는 코드·판별기·설정의 지문을 적고, 실측 실행 직전에 같은지 확인한다.

09 7장 5번(코드 동결), I-20. 실측 DDI 계산은 동결 확인이 통과한 뒤에만 한다.

    python -m rne.pipeline.freeze make --discriminator <파일> --settings <json> --out freeze/manifest.json
    python -m rne.pipeline.freeze verify freeze/manifest.json --discriminator <파일>

manifest: git 커밋, 작업 트리 깨끗함, 판별기 sha256, src/rne 아래 .py 파일마다 sha256과 전체 지문, 파이썬·주요 패키지 버전,
분석 설정(문턱 규칙, 기준선, 위치 경고 값, 증폭 없음 기준 등 — 사람이 정한 값).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "xlrd")


def sha256_file(path: str | Path) -> str:
    """파일 sha256(16진)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(root: str | Path = "src/rne") -> dict:
    """root 아래 .py 파일마다 sha256(줄바꿈을 LF로 맞춰 운영체제 차이를 없앤다)와 전체 지문."""
    files = {}
    for p in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        files[p.as_posix()] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    total = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode()).hexdigest()
    return dict(files=files, total=total)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def versions() -> dict:
    """파이썬과 주요 패키지 버전."""
    from importlib import metadata
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def make_manifest(discriminator: str | None, settings: dict, files: dict | None = None) -> dict:
    """동결 기록을 만든다. 작업 트리가 깨끗하지 않으면 clean=False로 적는다(동결로 쓰면 안 됨).

    files: {이름: 경로} — 배치표(웰 → 시료·조건), 시료표 등 결과를 바꾸는 입력 파일. sha256을 적는다.
    원자료(4·5차 판 내보내기)는 동결 뒤에 오므로 여기 넣을 수 없다 → rne.pipeline.resultday의 수령 기록(receipt)이 맡는다.
    """
    return dict(commit=_git("rev-parse", "HEAD"), clean=_git("status", "--porcelain", "--", "src") == "",
                discriminator=dict(path=discriminator, sha256=sha256_file(discriminator)) if discriminator else None,
                source=source_hashes(), versions=versions(), settings=settings,
                files={k: dict(path=str(v), sha256=sha256_file(v)) for k, v in (files or {}).items()})


def verify(manifest: dict, discriminator: str | None = None, settings: dict | None = None,
           files: dict | None = None) -> dict:
    """지금 상태가 동결 기록과 같은지. 반환: ok, problems(문장 목록).

    엄격하다: 기록에 설정이 있는데 설정을 주지 않으면 실패, 기록에 판별기가 없는데
    판별기를 주면 실패, 기록의 입력 파일(files)은 지금 경로(files 인자가 있으면 그 경로)의 sha256이 같아야 한다.
    """
    problems = []
    if not manifest.get("clean"):
        problems.append("동결 때 src 작업 트리가 깨끗하지 않았다 → 이 기록은 동결로 쓸 수 없다")
    now = source_hashes()
    if now["total"] != manifest["source"]["total"]:
        changed = sorted(k for k in set(now["files"]) | set(manifest["source"]["files"])
                         if now["files"].get(k) != manifest["source"]["files"].get(k))
        problems.append(f"소스가 바뀌었다({len(changed)}개): {', '.join(changed[:5])}")
    d = manifest.get("discriminator")
    if d:
        if discriminator is None:
            problems.append("판별기 파일을 주지 않았다")
        elif sha256_file(discriminator) != d["sha256"]:
            problems.append("판별기 sha256이 동결 기록과 다르다")
    elif discriminator is not None:
        problems.append("동결 기록에 없는 판별기를 주었다")
    if manifest.get("settings") is not None and settings is None:
        problems.append("분석 설정을 주지 않았다(동결 기록에 설정이 있다)")
    elif settings is not None and settings != manifest.get("settings"):
        problems.append("분석 설정이 동결 기록과 다르다")
    given = files or {}
    for name, rec in (manifest.get("files") or {}).items():
        path = given.get(name, rec["path"])
        if not Path(path).exists():
            problems.append(f"동결한 입력 파일이 없다: {name} ({path})")
        elif sha256_file(path) != rec["sha256"]:
            problems.append(f"입력 파일 sha256이 동결 기록과 다르다: {name} ({path})")
    extra = sorted(set(given) - set(manifest.get("files") or {}))
    if extra:
        problems.append(f"동결 기록에 없는 입력 파일: {', '.join(extra)}")
    v = versions()
    diff = {k: (manifest["versions"].get(k), v.get(k)) for k in v if v.get(k) != manifest["versions"].get(k)}
    if diff:
        problems.append(f"패키지 버전이 다르다(주의): {diff}")
    return dict(ok=not [p for p in problems if "(주의)" not in p], problems=problems)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--discriminator", default=None)
    m.add_argument("--settings", default=None, help="분석 설정 JSON 파일")
    m.add_argument("--file", action="append", default=[], help="이름=경로(배치표 등). 여러 번")
    m.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    v.add_argument("--discriminator", default=None)
    v.add_argument("--settings", default=None)
    v.add_argument("--file", action="append", default=[])
    a = ap.parse_args(argv)
    fl = dict(x.split("=", 1) for x in a.file)
    if a.cmd == "make":
        st = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
        man = make_manifest(a.discriminator, st, fl)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"동결 기록: {a.out} (커밋 {man['commit'][:7]}, 깨끗함 {man['clean']})")
        return 0 if man["clean"] else 2
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    st = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else None
    r = verify(man, a.discriminator, st, fl or None)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
