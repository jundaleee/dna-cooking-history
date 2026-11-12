"""PL3 코드 동결 시험."""
import json

from rne.pipeline import freeze as fz


def test_manifest_verify_detects_changes(tmp_path, monkeypatch):
    disc = tmp_path / "d.joblib"
    disc.write_bytes(b"model-v1")
    man = fz.make_manifest(str(disc), {"threshold": "plateau_0.10"})
    man["clean"] = True  # 시험에서는 작업 트리 상태와 무관하게
    assert fz.verify(man, str(disc), {"threshold": "plateau_0.10"})["ok"]
    disc.write_bytes(b"model-v2")
    r = fz.verify(man, str(disc), {"threshold": "plateau_0.10"})
    assert not r["ok"] and any("sha256" in p for p in r["problems"])
    disc.write_bytes(b"model-v1")
    r = fz.verify(man, str(disc), {"threshold": "plateau_0.20"})
    assert not r["ok"] and any("설정" in p for p in r["problems"])
    man2 = json.loads(json.dumps(man))
    k = next(iter(man2["source"]["files"]))
    man2["source"]["files"][k] = "0" * 64
    man2["source"]["total"] = "x"
    r = fz.verify(man2, str(disc), {"threshold": "plateau_0.10"})
    assert not r["ok"] and any("소스가 바뀌었다" in p for p in r["problems"])


def test_verify_is_strict_pl3(tmp_path):
    """설정 생략, 기록에 없는 판별기, 배치 파일 변경을 모두 실패로."""
    lay = tmp_path / "layout.csv"
    lay.write_text("well,sample_id\nA1,x\n", encoding="utf-8")
    man = fz.make_manifest(None, {"t": 1}, {"layout_P4": str(lay)})
    man["clean"] = True
    assert fz.verify(man, None, {"t": 1})["ok"]
    assert not fz.verify(man, None, None)["ok"]  # 설정을 주지 않음
    other = tmp_path / "d.pkl"
    other.write_bytes(b"retrained")
    assert not fz.verify(man, str(other), {"t": 1})["ok"]  # 기록에 없는 판별기
    lay.write_text("well,sample_id\nA1,y\n", encoding="utf-8")
    r = fz.verify(man, None, {"t": 1})
    assert not r["ok"] and any("입력 파일" in p for p in r["problems"])
