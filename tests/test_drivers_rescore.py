"""Audit 2026-10-04 fixes: pilot-scorer serialisation, rescore reference
handling, the v13 build gate, and the drivers' GPU-busy guard."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rescore_v5_protocol as rescore  # noqa: E402
import score_v13_pilot as pilot  # noqa: E402
from v13_build_status import build_status  # noqa: E402

GUARD_PATTERN = (r"experiments/finetuning/[A-Za-z]+/train\.py|scripts/run_eval_vllm\.py|"
                 r"run_mistral_seeds\.sh|run_schedule_[A-Za-z0-9_]+\.sh|run_v13_prog_pilot\.sh")


# --- score_v13_pilot: the set-not-serialisable crash of 2026-10-03 ----------

def test_public_block_is_json_serialisable():
    block = {"timebench": {"n": 3, "v5_pct": 50.0, "_correct": {"a", "b"}}}
    out = pilot.public(block)
    assert "_correct" not in out["timebench"]
    json.dumps(out)


def test_without_date_arith(tmp_path, monkeypatch):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"timebench": [
        "timebench-date_arith-x-1", "timebench-date_arith-x-2",
        "timebench-tracie-y-1", "timebench-tracie-y-2"]}))
    monkeypatch.setattr(pilot, "DEV_IDS", dev)
    r = pilot.without_date_arith({"n": 4, "_correct": {
        "timebench-date_arith-x-1", "timebench-date_arith-x-2", "timebench-tracie-y-1"}})
    assert r["n"] == 2 and r["date_arith_n"] == 2
    assert r["v5_pct"] == 50.0 and r["date_arith_pct"] == 100.0


def test_dev_filtered_dir_keeps_only_dev_ids(tmp_path, monkeypatch):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"time": ["t1"]}))
    monkeypatch.setattr(pilot, "DEV_IDS", dev)
    src = tmp_path / "preds.jsonl"
    src.write_text(json.dumps({"id": "t1"}) + "\n" + json.dumps({"id": "t2"}) + "\n")
    out = pilot.dev_filtered_dir({"time": src}, tmp_path / "f")
    lines = (out / "llama/time/finetuned/predictions.jsonl").read_text().splitlines()
    assert [json.loads(l)["id"] for l in lines] == ["t1"]


# --- rescore_v5_protocol: no silent fallback to the legacy reference --------

def test_missing_reference_label_is_a_hard_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rescore", "--reference", "no-such-arm",
                                      "--out", "/dev/null"])
    with pytest.raises(SystemExit) as e:
        rescore.main()
    assert "no-such-arm" in str(e.value)


def test_missing_reference_file_is_a_hard_error(monkeypatch):
    arms = {b: [("ghost-ref", "results/does/not/exist.jsonl")] for b in ("time", "timebench", "tram")}
    monkeypatch.setattr(rescore, "ARMS", arms)
    monkeypatch.setattr(sys, "argv", ["rescore", "--reference", "ghost-ref", "--out", "/dev/null"])
    with pytest.raises(SystemExit) as e:
        rescore.main()
    assert "missing" in str(e.value)


def test_model_family():
    assert rescore.model_family("zs-vllm-mistral") == "mistral"
    assert rescore.model_family("zsCoT-mistr") == "mistral"
    assert rescore.model_family("v11-best") == "llama"
    assert rescore.model_family("zs-vllm-pinned") == "llama"


def test_pilot_dir_is_not_an_arm():
    assert "results/hpo_v13_pilot/" in rescore._NOT_ARMS


# --- v13 build gate ---------------------------------------------------------

@pytest.mark.parametrize("manifest,final", [
    ({"provisional": True, "glm_rows": 0}, False),
    ({"provisional": True, "glm_rows": 0, "tpl_rows": 4519}, False),
    ({"provisional": False, "glm_rows": 0, "tpl_rows": 0}, False),
    ({"provisional": False, "glm_rows": 0, "tpl_rows": 4519}, True),
    ({"provisional": False, "glm_rows": 1200}, True),
])
def test_build_status(manifest, final):
    assert build_status(manifest).startswith("FINAL" if final else "PROVISIONAL")


def test_current_v13_build_is_still_gated():
    m = ROOT / "data/combined_80_20_v13/manifest.json"
    if not m.exists():
        pytest.skip("no v13 build")
    status = build_status(json.loads(m.read_text()))
    assert status.split()[0] in ("FINAL", "PROVISIONAL")


# --- drivers: GPU-busy guard ------------------------------------------------

def _guard_snippet(script: str) -> str:
    text = (ROOT / "scripts" / script).read_text()
    start = text.index("gpu_guard () {")
    end = text.index("\n}\n", start) + 3
    return text[start:end] + "gpu_guard\necho GUARD_PASSED\n"


@pytest.mark.parametrize("script", ["run_schedule_v13.sh", "run_v13_prog_pilot.sh",
                                    "run_mistral_seeds.sh"])
def test_gpu_guard_refuses_when_a_gpu_job_runs(script):
    fake = subprocess.Popen(["bash", "-c", 'exec -a "python scripts/run_eval_vllm.py --fake" sleep 30'])
    try:
        r = subprocess.run(["bash", "-c", _guard_snippet(script)],
                           capture_output=True, text=True, timeout=30)
    finally:
        fake.kill()
    assert r.returncode == 1 and "ABORT" in r.stdout and "GUARD_PASSED" not in r.stdout


def test_gpu_guard_passes_when_idle():
    busy = subprocess.run(["pgrep", "-af", GUARD_PATTERN], capture_output=True, text=True)
    if busy.stdout.strip():
        pytest.skip("a real GPU job is running")
    r = subprocess.run(["bash", "-c", _guard_snippet("run_schedule_v13.sh")],
                       capture_output=True, text=True, timeout=30)
    assert "GUARD_PASSED" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("script", ["run_schedule_v13.sh", "run_v13_prog_pilot.sh",
                                    "run_mistral_seeds.sh", "push_to_second_pc.sh"])
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)]).returncode == 0
